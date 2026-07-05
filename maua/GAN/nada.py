"""StyleGAN-NADA: text-guided domain adaptation of a StyleGAN2 generator.

Describe your source and target class. These describe the direction of change you're trying to
apply (e.g. "photo" -> "sketch", "dog" -> "the joker", or "dog" -> "avocado dog").

For changes that do not require drastic shape modifications, we recommend lambda_direction = 1.0,
lambda_patch = 0.0, lambda_global = 0.0. More drastic changes may require turning on the global
loss (and / or modifying the number of iterations).

As a rule of thumb:
- Style and minor domain changes ('photo' -> 'sketch') require ~200-400 iterations.
- Identity changes ('person' -> 'taylor swift') require ~150-200 iterations.
- Simple in-domain changes ('face' -> 'smiling face') may require as few as 50.
"""

import os
from argparse import Namespace

import torch
from tqdm import tqdm

from maua.GAN.ZSSGAN.model.ZSSGAN import ZSSGAN
from maua.GAN.ZSSGAN.utils.file_utils import save_images


def train(
    checkpoint_path,
    source_class,
    target_class,
    output_dir,
    name="nada",
    size=1024,
    batch=2,
    n_sample=28,
    lr=0.00025,
    iterations=500,
    output_interval=50,
    save_interval=250,
    truncation=0.7,
    lambda_direction=0.5,
    lambda_patch=0.5,
    lambda_global=0.25,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    args = Namespace(
        size=size,
        batch=batch,
        n_sample=n_sample,
        output_dir=output_dir,
        lr=lr,
        frozen_gen_ckpt=checkpoint_path,
        train_gen_ckpt=checkpoint_path,
        iter=iterations,
        source_class=source_class,
        target_class=target_class,
        target_img=None,
        lambda_direction=lambda_direction,
        lambda_patch=lambda_patch,
        lambda_global=lambda_global,
        phase=None,
        sample_truncation=truncation,
    )

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


def argument_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_path", help="path to the StyleGAN2 checkpoint to adapt")
    parser.add_argument("source_class", help="text describing the current domain, e.g. 'photo'")
    parser.add_argument("target_class", help="text describing the target domain, e.g. 'sketch'")
    parser.add_argument("--output-dir", default="output/nada", help="where to write samples/checkpoints")
    parser.add_argument("--name", default="nada", help="prefix for output files")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--n-sample", type=int, default=28)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--lambda-direction", type=float, default=0.5)
    parser.add_argument("--lambda-patch", type=float, default=0.5)
    parser.add_argument("--lambda-global", type=float, default=0.25)
    return parser


def main(args):
    train(
        checkpoint_path=args.checkpoint_path,
        source_class=args.source_class,
        target_class=args.target_class,
        output_dir=args.output_dir,
        name=args.name,
        size=args.size,
        batch=args.batch,
        n_sample=args.n_sample,
        lr=args.lr,
        iterations=args.iterations,
        output_interval=args.output_interval,
        save_interval=args.save_interval,
        truncation=args.truncation,
        lambda_direction=args.lambda_direction,
        lambda_patch=args.lambda_patch,
        lambda_global=args.lambda_global,
    )


if __name__ == "__main__":
    main(argument_parser().parse_args())
