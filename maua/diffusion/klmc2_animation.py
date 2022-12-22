# Copyright (c) 2022 Katherine Crowson <crowsonkb@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# pip install ftfy einops braceexpand requests transformers clip open_clip_torch omegaconf pytorch-lightning kornia k-diffusion ninja
# pip install -U torch torchvision
# pip install -U git+https://github.com/huggingface/huggingface_hub
# pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

import os
import subprocess
import sys
from concurrent import futures

import functorch
import huggingface_hub
import k_diffusion as K
import numpy as np
import torch
from maua.diffusion.load import load_diffusers
from omegaconf import OmegaConf
from requests.exceptions import HTTPError
from torch import nn
from tqdm.auto import trange

sys.path.extend(
    [
        "maua/submodules/stable_diffusion",
        "maua/submodules/stablediffusion",
        "maua/submodules/latent_diffusion",
        "maua/submodules/VQGAN",
    ]
)
from maua.submodules.stable_diffusion.ldm.util import instantiate_from_config

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
cpu = torch.device("cpu")
device = torch.device("cuda")


def download_from_huggingface(repo, filename):
    while True:
        try:
            return huggingface_hub.hf_hub_download(repo, filename, cache_dir="modelzoo/")
        except HTTPError as e:
            if e.response.status_code == 401:
                # Need to log into huggingface api
                huggingface_hub.interpreter_login()
                continue
            elif e.response.status_code == 403:
                # Need to do the click through license thing
                print(f"Go here and agree to the click through license on your account: https://huggingface.co/{repo}")
                input("Hit enter when ready:")
                continue
            else:
                raise e


def load_model_from_config(config, ckpt):
    if isinstance(ckpt, dict):
        pl_sd = ckpt
    else:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]

    config = OmegaConf.load(config)

    try:
        config["model"]["params"]["lossconfig"]["target"] = "torch.nn.Identity"
        print("Patched VAE config.")
    except KeyError:
        pass

    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model = model.to(cpu).eval().requires_grad_(False)
    return model


def load_models(sd_model_path=None):
    sd_model_path = sd_model_path or download_from_huggingface(
        "runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.ckpt"
    )
    vae_840k_model_path = download_from_huggingface(
        "stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt"
    )

    if os.path.isdir(sd_model_path):
        sd_model_path = load_diffusers(sd_model_path)
    model = (
        load_model_from_config(
            "maua/submodules/stable_diffusion/configs/stable-diffusion/v1-inference.yaml", sd_model_path
        )
        .half()
        .to(device)
    )

    vae_model = (
        load_model_from_config(
            "maua/submodules/latent_diffusion/models/first_stage_models/kl-f8/config.yaml",
            vae_840k_model_path,
        )
        .half()
        .to(device)
    )

    # Disable checkpointing as it is not compatible with the method
    for module in model.modules():
        if hasattr(module, "checkpoint"):
            module.checkpoint = False
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False

    return model, vae_model


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


@torch.no_grad()
def sample_mcmc_klmc2(
    model,
    x,
    sigma_min,
    sigma,
    sigma_max,
    n,
    h,
    gamma=2.0,
    alpha=0.0,
    tau=1.0,
    hvp_method="forward-functorch",
    extra_args=None,
    extra_args_non_mcmc=None,
    callback=None,
    disable=None,
):
    extra_args = {} if extra_args is None else extra_args
    extra_args_non_mcmc = {} if extra_args_non_mcmc is None else extra_args_non_mcmc
    s_in = x.new_ones([x.shape[0]])
    sigma = torch.tensor(sigma, device=x.device)
    sigmas = K.sampling.get_sigmas_karras(6, sigma_min, sigma.item(), device=x.device)[:-1]

    h = torch.tensor(h, device=x.device)
    gamma = torch.tensor(gamma, device=x.device)
    alpha = torch.tensor(alpha, device=x.device)
    tau = torch.tensor(tau, device=x.device)
    v = torch.randn_like(x) * sigma

    # Model helper functions
    def grad_fn(x, sigma):
        denoised = model(x, sigma * s_in, **extra_args)
        return (x - denoised) + alpha * x

    def hvp_fn_forward_functorch(x, sigma, v):
        jvp_fn = lambda v: functorch.jvp(grad_fn, (x, sigma), (v, torch.zeros_like(sigma)))
        grad, jvp_out = functorch.vmap(jvp_fn)(v)
        return grad[0], jvp_out

    def hvp_fn_reverse(x, sigma, v):
        vjps = []
        with torch.enable_grad():
            x_ = x.clone().requires_grad_()
            grad = grad_fn(x_, sigma)
            for k, item in enumerate(v):
                vjp_out = torch.autograd.grad(grad, x_, item, retain_graph=k < len(v) - 1)[0]
                vjps.append(vjp_out)
        return grad, torch.stack(vjps)

    def hvp_fn_zero(x, sigma, v):
        return grad_fn(x, sigma), torch.zeros_like(v)

    def hvp_fn_fake(x, sigma, v):
        return grad_fn(x, sigma), (1 + alpha) * v

    hvp_fns = {
        "forward-functorch": hvp_fn_forward_functorch,
        "reverse": hvp_fn_reverse,
        "zero": hvp_fn_zero,
        "fake": hvp_fn_fake,
    }

    hvp_fn = hvp_fns[hvp_method]

    # KLMC2 helper functions
    def psi_0(gamma, t):
        return torch.exp(-gamma * t)

    def psi_1(gamma, t):
        return -torch.expm1(-gamma * t) / gamma

    def psi_2(gamma, t):
        return (torch.expm1(-gamma * t) + gamma * t) / gamma**2

    def phi_2(gamma, t_):
        t = t_.double()
        out = (torch.exp(-gamma * t) * (torch.expm1(gamma * t) - gamma * t)) / gamma**2
        return out.to(t_)

    def phi_3(gamma, t_):
        t = t_.double()
        out = (torch.exp(-gamma * t) * (2 + gamma * t + torch.exp(gamma * t) * (gamma * t - 2))) / gamma**3
        return out.to(t_)

    shape_factor = 4
    fade_length = 60
    bias = np.concatenate(
        (
            np.linspace(shape_factor, 1 / shape_factor, fade_length // 2),
            np.linspace(1 / shape_factor, 1, fade_length // 2),
        )
    )

    x_norms, v_norms, xs, vs = [], [], [], []
    for i in trange(n + fade_length, disable=disable):
        if i <= fade_length:
            xs.append(x.clone().cpu())
            vs.append(v.clone().cpu())

        # Compute model outputs and sample noise
        x_trapz = torch.linspace(0, h, 1001, device=x.device)
        y_trapz = [fun(gamma, x_trapz) for fun in (psi_0, psi_1, phi_2, phi_3)]
        noise_cov = torch.tensor(
            [[torch.trapz(y_trapz[i] * y_trapz[j], x=x_trapz) for j in range(4)] for i in range(4)], device=x.device
        )
        noise_v, noise_x, noise_v2, noise_x2 = (
            torch.distributions.MultivariateNormal(x.new_zeros([4]), noise_cov).sample(x.shape).unbind(-1)
        )
        grad, (h2_v, h2_noise_v2, h2_noise_x2) = hvp_fn(x, sigma, torch.stack([v, noise_v2, noise_x2]))

        # DPM-Solver++(2M) refinement steps
        x_refine = x
        use_dpm = True
        old_denoised = None
        for j in range(len(sigmas) - 1):
            if j == 0:
                denoised = x_refine - grad
            else:
                denoised = model(x_refine, sigmas[j] * s_in, **extra_args_non_mcmc)
            dt_ode = sigmas[j + 1] - sigmas[j]
            if not use_dpm or old_denoised is None or sigmas[j + 1] == 0:
                eps = K.sampling.to_d(x_refine, sigmas[j], denoised)
                x_refine = x_refine + eps * dt_ode
            else:
                h_ode = sigmas[j].log() - sigmas[j + 1].log()
                h_last = sigmas[j - 1].log() - sigmas[j].log()
                fac = h_ode / (2 * h_last)
                denoised_d = (1 + fac) * denoised - fac * old_denoised
                eps = K.sampling.to_d(x_refine, sigmas[j], denoised_d)
                x_refine = x_refine + eps * dt_ode
            old_denoised = denoised
        if callback is not None:
            callback({"i": i % n, "denoised": x_refine})

        # Update the chain
        noise_std = (2 * gamma * tau * sigma**2).sqrt()
        v_next = (
            0
            + psi_0(gamma, h) * v
            - psi_1(gamma, h) * grad
            - phi_2(gamma, h) * h2_v
            + noise_std * (noise_v - h2_noise_v2)
        )
        x_next = (
            x
            + psi_1(gamma, h) * v
            - psi_2(gamma, h) * grad
            - phi_3(gamma, h) * h2_v
            + noise_std * (noise_x - h2_noise_x2)
        )
        v, x = v_next, x_next

        if i > n:
            ii = i - n
            fade_step_x = (xs[ii].to(x) - x) / (bias[ii] * (fade_length - ii))
            fade_step_v = (vs[ii].to(v) - v) / (bias[ii] * (fade_length - ii))
            x += fade_step_x
            v += fade_step_v
            x = x * (np.mean(x_norms) / x.norm())
            v = v * (np.mean(v_norms) / v.norm())
        else:
            x_norms.append(x.norm().item())
            v_norms.append(v.norm().item())

    x = x - grad
    return x


def generate_animation(prompt, cond_scale, n, fps, sigma, h, gamma, alpha, tau, hvp_method, model_path=None):
    model, vae_model = load_models(model_path)

    wrappers = {"eps": K.external.CompVisDenoiser, "v": K.external.CompVisVDenoiser}
    model_wrap = wrappers[model.parameterization](model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    uc = model.get_learned_conditioning(["watermarks, stock images, ugly"])
    c = model.get_learned_conditioning([prompt])
    extra_args = {"cond": c, "uncond": uc, "cond_scale": torch.tensor(cond_scale, device=device)}
    extra_args_non_mcmc = {"cond": c, "uncond": uc, "cond_scale": torch.tensor(7.5, device=device)}

    def save_image_fn(image, name, i):
        K.utils.to_pil_image(image).save(name)

    out_dir = f'output/klmc2_animation_{prompt.replace(" ", "_")}'
    os.makedirs(out_dir, exist_ok=True)
    torch.cuda.empty_cache()

    with torch.cuda.amp.autocast(), futures.ThreadPoolExecutor() as ex:

        def callback(info):
            i = info["i"]
            rgb = vae_model.decode(info["denoised"] / model.scale_factor)
            ex.submit(save_image_fn, rgb, f"{out_dir}/out_{i:05}.png", i)

        x = torch.randn([1, 4, 64, 64], device=device) * sigma_max

        # Initialize the chain
        print("Initializing the chain...")
        sigmas_pre = K.sampling.get_sigmas_karras(15, sigma, sigma_max, device=x.device)[:-1]
        x = K.sampling.sample_dpmpp_sde(model_wrap_cfg, x, sigmas_pre, extra_args=extra_args_non_mcmc)

        print("Actually doing the sampling...")
        sample_mcmc_klmc2(
            model_wrap_cfg,
            x,
            sigma_min,
            sigma,
            sigma_max,
            n,
            h,
            gamma=gamma,
            alpha=alpha,
            tau=tau,
            hvp_method=hvp_method,
            extra_args=extra_args,
            extra_args_non_mcmc=extra_args_non_mcmc,
            callback=callback,
        )

    subprocess.run(
        f"ffmpeg -y -r {fps} -i {out_dir}/out_%05d.png -crf 15 -preset veryslow -pix_fmt yuv420p {out_dir}.mp4",
        **dict(shell=True, stdout=sys.stdout, stderr=sys.stderr, cwd=os.getcwd(), env=os.environ),
    )


if __name__ == "__main__":
    # fmt:off
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--cond_scale", type=float, default=5.0, help="The strength of the conditioning on the prompt")
    parser.add_argument("--n", default=120, type=int, help="The number of frames to sample")
    parser.add_argument("--fps", default=20, type=int, help="Frames per second in output video")
    parser.add_argument("--sigma", default=0.75, type=float, help="The noise level to sample at")
    parser.add_argument("--h", default=0.2, type=float, help="Step size (range 0 to 1)")
    parser.add_argument("--gamma", default=0.5, type=float, help="Friction (2 is critically damped, lower -> smoother animation)")
    parser.add_argument("--alpha", default=1e-3, type=float, help="Quadratic penalty (weight decay) strength")
    parser.add_argument("--tau", default=1.0, type=float, help="Temperature (adjustment to the amount of noise added per step)")
    parser.add_argument("--hvp_method", default="fake", choices=["forward-functorch", "reverse", "fake", "zero"], help="The HVP method. `forward-functorch` and `reverse` provide real second derivatives. Compatibility, speed, and memory usage vary by model and xformers configuration. `fake` is very fast and low memory but inaccurate. `zero` (fallback to first order KLMC) is not recommended.")
    parser.add_argument("--model_path", default=None, type=str, help="Custom model checkpoint to load instead of Stable Diffusion v1.4")
    args = parser.parse_args()
    # fmt:on

    generate_animation(**vars(args))
