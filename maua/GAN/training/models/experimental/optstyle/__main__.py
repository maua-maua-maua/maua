import argparse
from pathlib import Path
from typing import Generator as PythonGenerator
from typing import List

import torch
import torchvision as tv
from numpy import sqrt
from torch.nn.functional import one_hot
from torchvision.transforms.functional import to_tensor

from ......ops.io import tensor2img
from .....sampling import sample_latents
from .....wrappers import MauaGenerator, get_generator_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_images(
    G: MauaGenerator,
    seeds: List[int],
    class_idx: int,
    truncation: torch.Tensor,
    latent_sampling: str,
    langevin_critic: str,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    batch_size: int,
) -> PythonGenerator[torch.Tensor, None, None]:

    # G = torch.jit.trace(G, torch.randn((batch_size, G.z_dim), device=device))
    # G = torch.jit.optimize_for_inference(G)

    class_conditioning = one_hot(class_idx, num_classes=G.c_dim).to(device) if class_idx is not None else None

    latents = sample_latents(G, seeds, batch_size, truncation, latent_sampling, langevin_critic)

    with torch.inference_mode():
        for i in range(0, len(latents), batch_size):
            latent_z = latents[i : i + batch_size].to(device)
            imgs = G.forward(
                latent_z,
                class_conditioning,
                translation=translation,
                rotation=rotation,
                truncation=truncation if latent_sampling == "standard" else 1,
            )
            for img in imgs[:, None]:
                yield img.add(1).div(2).cpu()


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True, type=str, help="Path to .pkl/.pt file containing the model to use")
    parser.add_argument("--architecture", default="stylegan3", type=str, choices=["stylegan2", "stylegan3"], help="The architecture of the model")
    parser.add_argument("--seeds", default="42", type=str, help="Comma separated list of seeds to generate images for. Use a dash to specify a range: e.g. '1,3,5,6-10'")
    parser.add_argument("--class_idx", default=None, type=int, help="Index of class to generate (only applicable to conditional models)")
    parser.add_argument("--truncation", default=1.0, type=float, help="Latent truncation value. Lower values give higher quality with less diversity; higer values vice versa")
    parser.add_argument("--latent_sampling", default='standard', choices=['standard', 'langevin', 'polarity', 'jacobian'], type=str, help="Strategy used to sample latents, for more details see maua/GAN/sampling/README.md")
    parser.add_argument("--langevin_critic", default='discriminator', type=str, help="Critic to use for langevin latent sampling. Either 'discriminator' for standard DDLS or a text prompt for CLIP-guided langevin sampling")
    parser.add_argument("--translation", default=None, type=str, help="x,y offset values for StyleGAN3's input transformation matrix (translates image in latent space)")
    parser.add_argument("--rotation", default=None, type=float, help="Rotation value for StyleGAN3's input transformation matrix (rotates image in latent space)")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--out_size", default="1024,1024", type=str, help="Desired width,height of output image: e.g. 1920,1080 or 720,1280")
    parser.add_argument("--resize_strategy", default="stretch", type=str, help="Strategy used to resize (in feature space) to achieve desired output resolution")
    parser.add_argument("--resize_layer", default=0, choices=list(range(18)), type=int, help="Which layer in the network to perform resizing at. Higher values are closer to resizing output pixels directly. Lower values have larger rounding increments (i.e. less flexible possible output sizes)")
    parser.add_argument("--grid", action='store_true', help="Whether to output images together as a grid, rather than image by image")
    parser.add_argument("--out_dir", default="./output/", type=str, help="Directory to output images in")
    args = parser.parse_args()
    
    seeds = sum([([int(seed)] if not "-" in seed else list(range(int(seed.split("-")[0]), int(seed.split("-")[1])))) for seed in args.seeds.split(",")], [])
    # fmt: on

    translation = (
        torch.tensor([float(s) for s in args.translation.split(",")]) if args.translation is not None else None
    )
    rotation = torch.tensor(args.rotation) if args.rotation is not None else None
    out_size = tuple(int(s) for s in args.out_size.split(","))

    G_cls = get_generator_class(args.architecture)
    G = G_cls(
        model_file=args.model_file, output_size=out_size, strategy=args.resize_strategy, layer=args.resize_layer
    ).to(device)

    import matplotlib.pyplot as plt
    from PIL import Image
    from resize_right import resize

    from .optimal_transport import sliced_optimal_transport

    im = Image.open("/home/hans/HDDs/datasets/2020:11:11:11:22:33/tumblr_pnzb8cidoo1r20fq5o1_1280.jpg")
    im = to_tensor(im).cuda().mean(0)[None, None, ...]

    def info(x, l):
        print(l, tuple(x.shape), f"{x.min().item():.2f}", f"{torch.median(x).item():.2f}", f"{x.max().item():.2f}")

    for name, layer in list(G.synthesizer.G_synth.named_children())[1:-1]:

        def hook(mod, inp, out, name=name):
            print(name)
            info(out, "in")
            # plt.hist(out.flatten().cpu().numpy(), bins=30, alpha=1 / 3, label="in")
            img = resize(im, out_shape=out.shape[2:]).tile(1, out.shape[1], 1, 1)
            img = img * (out.max() - out.min()) + out.min()
            info(img, "im")
            # plt.hist(img.flatten().cpu().numpy(), bins=30, alpha=1 / 3, label="im")
            # out = (99 * out + img) / 100
            # out = out * img
            out = sliced_optimal_transport(out, img)
            info(out, "out")
            # plt.hist(out.flatten().cpu().numpy(), bins=30, alpha=1 / 3, label="out")
            # plt.legend()
            # plt.tight_layout()
            # plt.show()
            # plt.close()
            return out

        layer.register_forward_hook(hook)

    imgs = generate_images(
        G=G,
        seeds=seeds,
        class_idx=args.class_idx,
        truncation=args.truncation,
        latent_sampling=args.latent_sampling,
        langevin_critic=args.langevin_critic,
        translation=translation,
        rotation=rotation,
        batch_size=args.batch_size,
    )

    out_name = f'{Path(args.model_file.replace("/network-snapshot", "")).stem}'
    if out_size[0] != G.res or out_size[1] != G.res:
        out_name += f"_{args.resize_strategy}@{args.resize_layer}_{out_size[0]}x{out_size[1]}"
    if args.latent_sampling != "standard":
        out_name += f"_{args.latent_sampling}"
        if args.latent_sampling == "langevin":
            out_name += f":{args.langevin_critic}"
    if args.truncation != 1.0:
        out_name += f"_trunc{args.truncation}"

    if args.grid:
        tv.utils.save_image(
            torch.cat(list(imgs)),
            f"{args.out_dir}/seeds{args.seeds}_{out_name}.jpg",
            nrow=round(sqrt(len(seeds)) * 4 / 3),
        )
    else:
        for seed, img in zip(seeds, imgs):
            img = tensor2img(img)
            img.save(f"{args.out_dir}/seed{seed}_{out_name}.jpg")
