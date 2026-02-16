"""
Describe your source and target class. These describe the direction of change you're trying to apply (e.g. "photo" to "sketch", "dog" to "the joker" or "dog" to "avocado dog").

For changes that do not require drastic shape modifications, we reccomend lambda_direction = 1.0, lambda_patch = 0.0, lambda_global = 0.0.

More drastic changes may require turning on the global loss (and / or modifying the number of iterations).


As a rule of thumb:
- Style and minor domain changes ('photo' -> 'sketch') require ~200-400 iterations.
- Identity changes ('person' -> 'taylor swift') require ~150-200 iterations.
- Simple in-domain changes ('face' -> 'smiling face') may require as few as 50.
"""

import os
from argparse import Namespace

import torch
from PIL import Image
from tqdm import tqdm

from maua.GAN.ZSSGAN.model.ZSSGAN import ZSSGAN
from maua.GAN.ZSSGAN.utils.file_utils import save_images

output_dir = "/home/hans/modelzoo/koanGAN/nada"

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

source_class = "abstract city landscape art"
target_class = "futuristic cityscape at sunset with a beautiful red sky, detailed, high contrast"
target_img = Image.open("/home/hans/HDDs/datasets/koan/01 Embers from Chaos.jpg").convert("RGB")

lambda_direction = 0.5
lambda_patch = 0.5
lambda_global = 0.25

training_iterations = 500
output_interval = 50
save_interval = 250
truncation = 0.7

checkpoint_path = "/home/hans/modelzoo/koanGAN/select/quoxal-koancept-sylvaleonsce-blend.pt"
name = "embers-from-chaos"

training_args = {
    "size": 1024,
    "batch": 2,
    "n_sample": 28,
    "output_dir": output_dir,
    "lr": 0.00025,
    "frozen_gen_ckpt": checkpoint_path,
    "train_gen_ckpt": checkpoint_path,
    "iter": training_iterations,
    "source_class": source_class,
    "target_class": target_class,
    "target_img": None,  # target_img,
    "lambda_direction": lambda_direction,
    "lambda_patch": lambda_patch,
    "lambda_global": lambda_global,
    "phase": None,
    "sample_truncation": truncation,
}
args = Namespace(**training_args)

os.makedirs(output_dir, exist_ok=True)

net = ZSSGAN(args)

g_reg_ratio = 4 / 5
g_optim = torch.optim.Adam(
    net.generator_trainable.parameters(), lr=args.lr * g_reg_ratio, betas=(0**g_reg_ratio, 0.99**g_reg_ratio)
)

fixed_z = torch.randn(args.n_sample, 512, device=device)

for i in tqdm(range(args.iter + 1)):
    sample_z = torch.randn(args.batch, 512, device=device)
    [sampled_src, sampled_dst], [cycle_dst, cycle_src], clip_loss, cycle_loss = net([sample_z])

    net.zero_grad()
    clip_loss.backward()
    g_optim.step()

    if i % output_interval == 0:
        with torch.no_grad():
            samples = []
            for ns in range((args.n_sample + 7) // 8):
                [sampled_src, sampled_dst], [cycle_dst, cycle_src], clip_loss, cycle_loss = net(
                    [fixed_z[ns * 8 : (ns + 1) * 8]], truncation=args.sample_truncation
                )
                samples.append(sampled_dst.cpu())
            samples = torch.cat(samples)
            save_images(samples, args.output_dir, f"{name}_{str(i).zfill(4)}", 7)

    if (i + 1) % save_interval == 0:
        state_dict = net.generator_trainable.state_dict()
        state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items()}
        torch.save(state_dict, f"{output_dir}/{name}_{str(i + 1).zfill(4)}.pt")
