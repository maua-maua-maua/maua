import sys
from pathlib import Path

import torch
from kornia.filters import gaussian_blur2d
from maua.submodules.k_diffusion.k_diffusion.utils import to_pil_image

from ..prompt import ImagePrompt, TextPrompt
from .processors.stable import StableDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_rotation(N):
    """
    Draws random N-dimensional rotation matrix (det = 1, inverse = transpose) from the special orthogonal group
    from https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L3309
    """
    H = torch.eye(N, device=device)
    D = torch.empty((N,), device=device)
    for n in range(N - 1):
        x = torch.randn(N - n, device=device)
        norm2 = x @ x
        x0 = x[0].item()
        D[n] = torch.sign(x[0]) if x[0] != 0 else 1
        x[0] += D[n] * torch.sqrt(norm2)
        x /= torch.sqrt((norm2 - x0**2 + x[0] ** 2) / 2.0)
        H[:, n:] -= torch.outer(H[:, n:] @ x, x)
    D[-1] = (-1) ** (N - 1) * D[:-1].prod()
    H = (D * H.T).T
    return H


def match_histogram(target, source, mode="chol", eps=1e-2):
    """from https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36"""

    target = target.permute(0, 3, 1, 2)  # -> b, c, h, w
    source = source.permute(0, 3, 1, 2)

    mu_t = target.mean((2, 3), keepdim=True)
    hist_t = (target - mu_t).view(target.size(1), -1)  # [c, b * h * w]
    cov_t = hist_t @ hist_t.T / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0], device=device)

    mu_s = source.mean((2, 3), keepdim=True)
    hist_s = (source - mu_s).view(source.size(1), -1)
    cov_s = hist_s @ hist_s.T / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0], device=device)

    if mode == "chol":
        chol_t = torch.linalg.cholesky(cov_t)
        chol_s = torch.linalg.cholesky(cov_s)
        matched = chol_s @ torch.inverse(chol_t) @ hist_t

    elif mode == "pca":
        eva_t, eve_t = torch.linalg.eigh(cov_t, UPLO="U")
        Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
        eva_s, eve_s = torch.linalg.eigh(cov_s, UPLO="U")
        Qs = eve_s @ torch.sqrt(torch.diag(eva_s)) @ eve_s.T
        matched = Qs @ torch.inverse(Qt) @ hist_t

    elif mode == "sym":
        eva_t, eve_t = torch.linalg.eigh(cov_t, UPLO="U")
        Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
        Qt_Cs_Qt = Qt @ cov_s @ Qt
        eva_QtCsQt, eve_QtCsQt = torch.linalg.eigh(Qt_Cs_Qt, UPLO="U")
        QtCsQt = eve_QtCsQt @ torch.sqrt(torch.diag(eva_QtCsQt)) @ eve_QtCsQt.T
        matched = torch.inverse(Qt) @ QtCsQt @ torch.inverse(Qt) @ hist_t

    matched = matched.view(*target.shape) + mu_s

    return matched.permute(0, 2, 3, 1)  # -> b, h, w, c


def sliced_optimal_transport(target, source, hist_mode="chol", iterations=8):
    """from https://github.com/JCBrouwer/OptimalTextures/blob/main/optex.py#L121"""

    target, source = target.permute(0, 2, 3, 1), source.permute(0, 2, 3, 1)  # -> b, h, w, c

    for _ in range(iterations):

        rotation = random_rotation(target.shape[-1])

        rotated_output = target @ rotation
        rotated_style = source @ rotation

        matched_output = match_histogram(rotated_output, rotated_style, mode=hist_mode)

        target = matched_output @ rotation.T  # rotate back to normal

    return target.permute(0, 3, 1, 2)  # -> b, c, h, w


class OutpaintingStableDiffusion(StableDiffusion):
    @torch.inference_mode()
    def outpaint(self, img, prompts, t_start=0, t_end=1, verbose=True, latent=False):
        if not hasattr(self, "sigmas"):
            raise NotImplementedError("p, ddim, and plms samplers are not supported with outpainting yet...")

        sigmas = self.get_sigmas(t_start, t_end)
        steps = len(sigmas) - 1

        prompts = [p.to(img) for p in prompts]
        [gm.set_targets(prompts) for gm in self.grad_modules]
        cond, uncond = self.conditioning(prompts)
        cond_shape = (img.shape[0], cond.shape[1], cond.shape[2])
        cond_info = {"cond": cond.expand(cond_shape), "uncond": uncond.expand(cond_shape), "cond_scale": self.cfg_scale}

        x = img if latent else self.encode(img)
        b, c, h, w = x.shape
        X = torch.randn(b, c, h * 2, w * 2, device=x.device)
        X = sliced_optimal_transport(X, x)
        X[..., h // 2 : -h // 2, w // 2 : -w // 2] = x

        Y = torch.randn_like(X)
        Y = sliced_optimal_transport(Y, x)

        mask = torch.ones_like(X).mul(-0.1)  # -0.1 to ensure gaussian blur pulls 0 vals into original img a litle
        mask[..., h // 2 : -h // 2, w // 2 : -w // 2] = 1
        mask = gaussian_blur2d(mask, kernel_size=(15, 15), sigma=(5, 5))
        mask = mask.clamp(0, 1)

        XY = X * mask + Y * (1 - mask)

        with torch.autocast(self.device), self.model.ema_scope():

            def outpainting_callback(it):
                fade = 1 - it["i"] / steps  # more iterations --> less mask value --> less XY influence
                mask_i = fade * mask
                xy = XY + torch.randn_like(XY) * sigmas[it["i"]]
                x_new = xy * mask_i + it["x"] * (1 - mask_i)
                it["x"].set_(x_new)

            out = self.sample_fn(
                self.model_fn,
                XY + torch.randn_like(XY) * sigmas[0],
                sigmas,
                extra_args=cond_info,
                disable=not verbose,
                callback=outpainting_callback,
            )

            out = out if latent else self.decode(out)

        return out.float()


if __name__ == "__main__":
    init, text = sys.argv[1], sys.argv[2]
    out_name = text.replace(" ", "_")

    diffusion = OutpaintingStableDiffusion(sampler="euler_ancestral")

    if init == "none":
        img = diffusion.forward(torch.randn(1, 3, 512, 512, device="cuda"), [TextPrompt(text)], t_start=0)
        to_pil_image(img.squeeze()).save(f"output/{out_name}.png")
    else:
        out_name = f"{Path(init).stem}_{out_name}"
        img = ImagePrompt(path=init).img

    img = diffusion.outpaint(img, [TextPrompt(text)], t_start=0.4)

    to_pil_image(img.squeeze()).save(f"output/outpainted_{out_name}.png")
