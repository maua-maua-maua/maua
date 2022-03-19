import gc
import os
import traceback
from functools import partial
from math import ceil, sqrt

import torch
import torchvision as tv
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.transforms.functional import resize
from tqdm import tqdm

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._set_cublas_allow_tf32(True)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.jit.optimized_execution(True)
torch.jit.fuser("fuser2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_critic(how, model_file):
    if how == "discriminator":
        try:
            from ..nv import dnnlib, legacy

            with dnnlib.util.open_url(model_file) as f:
                critic = legacy.load_network_pkl(f)["D"].eval().to(device)
            critic = partial(critic, c=None)
        except:
            print()
            traceback.print_exc()
            raise ValueError(
                f"\n\nFailed to load discriminator from {model_file}!\nAt the moment only checkpoints which can be loaded by the official StyleGAN3 repo are supported.\n"
            )
    else:
        import clip
        from torchvision.transforms import Normalize
        from torchvision.transforms.functional import resize

        CLIP, _ = clip.load("ViT-B/32", device=device, jit=True)
        normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        text_features = CLIP.encode_text(clip.tokenize(how).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

        def critic(imgs):
            imgs = normalize(resize(imgs, size=224, antialias=True))
            image_features = CLIP.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            logits = 20 * similarities.squeeze()
            print(logits.mean())
            return logits

    return critic


def langevin_with_critic(
    G, zs, critic, bs=8, rate=0.1, noise_std=1, decay=0.25, decay_steps=200, steps=1000, log_intermediate=True
):
    """
    Discriminator Driven Latent Sampling from "Your GAN is Secretly an Energy-based Model" by Che et al.
    https://arxiv.org/abs/2003.06060

    Args:
        G (torch.nn.Module): Generator
        n_samples (int): Number of latents to return
        critic (str): What to use as critic function (either 'discriminator' for DDLS or a text prompt for CLIP-guided sampling)
        bs (int): Batch size
        rate (float): an initial update rate for langevin sampling
        noise_std (float): standard deviation of a gaussian noise used in langevin sampling
        decay (float): decay strength for rate and noise_std
        decay_steps (int): rate and noise_std decrease every decay_steps
        steps (int): total steps of langevin sampling
    """

    scaler = 1.0
    apply_decay = decay > 0 and decay_steps > 0

    mean = torch.zeros(G.z_dim, device=device)
    prior_std = torch.eye(G.z_dim, device=device)
    lgv_std = prior_std * noise_std
    prior = MultivariateNormal(loc=mean, covariance_matrix=prior_std)
    lgv_prior = MultivariateNormal(loc=mean, covariance_matrix=lgv_std)

    l = len(zs)
    new_zs = torch.randn((bs * ceil(l / bs), zs.shape[1]), device=device)
    new_zs[:l] = zs
    zs = new_zs
    zs = zs.to(device)

    how = critic
    critic = prepare_critic(critic, G.model_file)

    G = torch.jit.trace(G, torch.randn((bs, G.z_dim), device=device))
    G = torch.jit.optimize_for_inference(G)

    if log_intermediate:
        ldir = f"output/langevin_{how}"
        if not os.path.exists(ldir):
            os.makedirs(ldir, exist_ok=True)

    for i in tqdm(range(steps), desc="Langevin sampling...", smoothing=0.01):
        with torch.no_grad():
            fake_logits = []
            for b in range(0, zs.shape[0], bs):
                imgs = G(zs[b : b + bs])
                logits = critic(imgs)
                fake_logits.append(logits)
            fake_logits = torch.stack(fake_logits).flatten()

        zs.requires_grad_()
        energy = -prior.log_prob(zs) - fake_logits
        z_grads = torch.autograd.grad(
            outputs=energy,
            inputs=zs,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        zs = zs - 0.5 * rate * z_grads + rate**0.5 * lgv_prior.sample([zs.shape[0]]) * scaler

        if apply_decay and (i + 1) % decay_steps == 0:
            rate *= decay
            scaler *= decay

        if log_intermediate:
            with torch.inference_mode():
                imgs = []
                for j in range(0, len(zs), bs):
                    imgs.append(resize(G(zs[j : j + bs].to(device)).add(1).div(2), size=256, antialias=True).cpu())
                tv.utils.save_image(torch.cat(imgs), f"{ldir}/{i:04}.jpg", nrow=round(sqrt(len(zs)) * 4 / 3))

    del critic, imgs, fake_logits, energy, z_grads
    gc.collect()
    torch.cuda.empty_cache()

    return zs[:l]
