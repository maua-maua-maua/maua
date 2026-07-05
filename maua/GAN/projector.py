# %%
import math
import os
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_32
from torchvision.transforms.functional import five_crop, resize, to_tensor
from tqdm import tqdm, trange

from maua.GAN.wrappers import get_generator_class

np.set_printoptions(precision=2, suppress=True)
# %%


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = np.minimum(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * np.minimum(1, t / rampup)
    return initial_lr * lr_ramp


def make_image(tensor):
    return (
        tensor.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    )


class LatentImageDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(os.listdir(f"{projector_set}/")) // 2

    def __getitem__(self, idx):
        w = torch.load(f"{projector_set}/{idx:05}.pt").squeeze()
        img = to_tensor(Image.open(f"{projector_set}/{idx:05}.jpg").resize((224, 224), Image.Resampling.LANCZOS))
        return w, img


def _log_cosh(x):
    return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)


def log_cosh_loss(y_pred, y_true):
    return torch.mean(_log_cosh(y_pred - y_true))


def crop(img):
    w = img.shape[-1]
    crops = five_crop(img, w // 2)

    res = [
        resize(img, size=w // 4, antialias=True),
        *[resize(c, size=w // 4, antialias=True) for c in crops],
    ]
    for crop in crops:
        res += list(five_crop(crop, w // 4))

    return torch.cat(res, dim=1)


def stats(latent):
    return torch.cat((
        torch.quantile(latent, q=0.01, dim=2).mean(0),
        torch.quantile(latent, q=0.10, dim=2).mean(0),
        torch.quantile(latent, q=0.25, dim=2).mean(0),
        torch.quantile(latent, q=0.50, dim=2).mean(0),
        torch.quantile(latent, q=0.75, dim=2).mean(0),
        torch.quantile(latent, q=0.90, dim=2).mean(0),
        torch.quantile(latent, q=0.99, dim=2).mean(0),
        torch.mean(latent, dim=2).mean(0),
        torch.std(latent, dim=2).mean(0),
    ))


# %%
def noimomes(noise):
    return torch.stack((
        torch.quantile(noise, q=0.01),
        torch.quantile(noise, q=0.10),
        torch.quantile(noise, q=0.25),
        torch.quantile(noise, q=0.50),
        torch.quantile(noise, q=0.75),
        torch.quantile(noise, q=0.90),
        torch.quantile(noise, q=0.99),
        torch.mean(noise),
        torch.std(noise),
    ))


def project(model_file, file, out_dir="output", device="cuda", use_vit=False, steps=5000):
    os.makedirs(out_dir, exist_ok=True)
    noitareget_moms = noimomes(torch.randn(1, 3, 1024, 1024, device=device))

    # %%
    name = Path(model_file).stem
    g_ema = get_generator_class("stylegan2")(model_file=model_file)
    g_ema.eval()
    g_ema = g_ema.to(device)

    def display(im: Image.Image):
        im.save(f"{out_dir}/{name}_{Path(file).stem}.jpg")

    def latent_noise(latent, strength):
        noise = g_ema.mapper(torch.randn_like(latent.mean(1))) * strength
        return latent + noise

    def save(path):
        result_file = {"latent": latent_in, "noise": noises}
        torch.save(result_file, f"{path}.pt")
        img_gen = g_ema.synthesizer.forward(latent_path[-1], **{f"noise{n}": noise for n, noise in enumerate(noises)})
        Image.fromarray(make_image(img_gen).squeeze()).save(f"{path}.jpg")

    # %%
    if use_vit:
        train_vit = False
        if train_vit:
            n = 10000
            batch_size = 32

            projector_set = f"{out_dir}/projector_{name}/"
            os.makedirs(projector_set, exist_ok=True)

            with torch.no_grad():
                i = 0
                for _ in trange(n // batch_size):
                    zs = torch.randn(batch_size, 512, device=device)
                    ws = g_ema.mapper(zs)
                    imgs = g_ema.synthesizer.forward(ws)
                    for w, img in zip(ws, imgs.unsqueeze(1)):
                        torch.save(w.cpu(), f"{projector_set}/{i:05}.pt")
                        Image.fromarray(make_image(img).squeeze()).save(f"{projector_set}/{i:05}.jpg")
                        i += 1

            # %%
            vit = vit_b_32(weights="DEFAULT")
            vit.heads = torch.nn.Sequential(
                torch.nn.Linear(vit.hidden_dim, vit.hidden_dim * 2),
                torch.nn.LayerNorm(vit.hidden_dim * 2),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(vit.hidden_dim * 2, np.prod(w.shape)),
                torch.nn.LeakyReLU(0.2),
            )
            vit = vit.to(device)

            # %%
            epochs = 32
            batch_size = 32
            dataloader = torch.utils.data.DataLoader(
                LatentImageDataset(), batch_size=batch_size, num_workers=24, shuffle=True
            )

            optimizer = torch.optim.Adam(vit.heads.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader), eta_min=5e-6)

            i = 0
            for e in trange(epochs):
                with tqdm(dataloader, unit_scale=batch_size) as pbar:
                    for ws, imgs in pbar:
                        ws, imgs = ws.to(device), imgs.to(device)
                        ws_p = vit(imgs).reshape(ws.shape)
                        main_loss = log_cosh_loss(ws_p, ws)
                        mean_loss = log_cosh_loss(ws_p.mean(-1).mean(-1), ws.mean(-1).mean(-1))
                        loss = main_loss + mean_loss
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        if i % (len(dataloader) // 4) == 0:
                            num = 6
                            img_w = make_image(g_ema.synthesizer.forward(ws[:num]))
                            img_p = make_image(g_ema.synthesizer.forward(ws_p[:num]))
                            grid = Image.fromarray(
                                np.concatenate((np.concatenate(img_w, axis=1), np.concatenate(img_p, axis=1)), axis=0)
                            ).resize((224 * num, 224 * 2))
                            # clear_output()
                            display(grid)
                            torch.save(vit, f"{out_dir}/{name}-vit-latent-encoder.pt")

                        pbar.set_description(
                            f"Epoch: {e}/{epochs}    Main: {main_loss.item():.4f}   Mean: {main_loss.item():.4f}    "
                            f"LR: {scheduler.get_last_lr()[0]:.3g}   LatStats:"
                            f"({ws.min().item():.2f} {ws.mean().item():.2f} {ws.max().item():.2f}) "
                            f"({ws_p.min().item():.2f} {ws_p.mean().item():.2f} {ws_p.max().item():.2f})"
                        )
                        i += 1
        # %%
        else:
            vit = torch.load(f"{out_dir}/{name}-vit-latent-encoder.pt")

    # %%
    size = g_ema.synthesizer.G_synth.img_resolution  # match the target to the generator's output resolution
    lr = 0.1
    lr_rampup = 0.01
    lr_rampdown = 0.6
    noise_scale = 0.01
    noise_ramp = 0.75
    step = steps
    tv = 0
    mse = 1
    latstat = 5
    l_noimome = 25
    n_mean_latent = 2**14
    use_crops = True


    transform = transforms.Compose([
        transforms.Resize(size, antialias=True),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    with torch.no_grad():
        real = transform(Image.open(file).convert("RGB")).unsqueeze(0).to(device)
        if use_crops:
            real = crop(real)

        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_sample = g_ema.mapper(noise_sample)

        latent_mean = latent_sample.mean(0)
        latent_std = ((latent_sample - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        latent_stats = stats(latent_sample)

        percept = lpips.LPIPS(net="vgg").to(device)

        noises = []
        for l, layer in enumerate(g_ema.synthesizer.layer_names[1:]):
            _, block, conv = layer.split(".")
            synth_layer = getattr(g_ema.synthesizer.G_synth.bs[int(block)], conv)
            h, w = synth_layer.noise_const.shape[-2], synth_layer.noise_const.shape[-1]
            noises.append(torch.randn(real.shape[0], 1, h, w, device=device))

        if use_vit:
            latent_in = vit(resize(real, 224, antialias=True)).reshape(latent_mean.shape).to(device)
        else:
            latent_in = latent_mean.unsqueeze(0).clone()
        latent_in += 0.01 * torch.randn(size=(1, latent_in.shape[1], 1)).to(latent_in)

    latent_in.requires_grad = True
    for noise in noises:
        noise.requires_grad = True

    optimizer = torch.optim.Adam([latent_in] + noises, lr=lr)

    save_path = f"{out_dir}/{name}-projected-{Path(file).stem}"
    # %%
    pbar = tqdm(range(step))
    latent_path = []

    for i in pbar:
        t = i / step
        lr_ = get_lr(t, lr, lr_rampdown, lr_rampup)
        optimizer.param_groups[0]["lr"] = lr_
        noise_strength = latent_std * noise_scale * max(0, 1 - t / noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen = g_ema.synthesizer.forward(latent_n, **{f"noise{n}": noise for n, noise in enumerate(noises)})

        batch, channel, height, width = img_gen.shape

        if use_crops:
            img_gen = crop(img_gen)

        p_loss = sum([percept(ig, ir).sum() for ig, ir in zip(img_gen.split(3, dim=1), real.split(3, dim=1))])
        mse_loss = log_cosh_loss(img_gen, real)
        stat_loss = log_cosh_loss(latent_stats, stats(latent_n))
        noimome_vals = [noimomes(n) for n in noises]
        noimome_loss = sum([log_cosh_loss(n, noitareget_moms) for n in noimome_vals])

        loss = p_loss + mse * mse_loss + latstat * stat_loss + l_noimome * noimome_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        noise_normalize_(noises)

        stat_diff = stats(latent_n) - latent_stats
        noimome_diff = torch.cat([n - noitareget_moms for n in noimome_vals])
        pbar.set_description(
            f"perceptual: {p_loss.item():.4f}; mse: {mse_loss.item():.4f}; lr: {lr_:.2f}; "
            f"stat reg {stat_loss.item():.4f}; stats diff: ({stat_diff.min():.2f} {stat_diff.mean():.2f} {stat_diff.max():.2f}); "
            f"noimome reg {noimome_loss.item():.4f}; noimomes diff: ({noimome_diff.min():.2f} {noimome_diff.mean():.2f} {noimome_diff.max():.2f}) "
        )

        if i % 100 == 0:
            latent_path.append(latent_in.detach().clone())
            save(save_path)
            print()

    pbar.close()

    # %%
    with torch.no_grad():
        save_dict = torch.load(f"{save_path}.pt")
        latent = save_dict["latent"]
        img = g_ema.synthesizer.forward(latent_n)
        display(Image.fromarray(make_image(img).squeeze()))
        print(stats(latent).cpu().numpy())
        print(latent_stats.cpu().numpy())


if __name__ == "__main__":
    project(model_file=sys.argv[1], file=sys.argv[2])
