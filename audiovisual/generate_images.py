import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.functional import one_hot

from .util import tensor2imgs
from .wrappers import get_generator_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @torch.jit.script
def generate_images(
    model_file: str,
    architecture: str,
    seeds: List[int],
    class_idx: int,
    truncation: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
    out_size: Tuple[int, int],
    resize_strategy: str,
    resize_layer: int,
) -> torch.Tensor:

    mapper, synthesizer = get_generator_classes(architecture)
    G_map = mapper(model_file).to(device)
    G_synth = synthesizer(model_file, out_size, resize_strategy, resize_layer).to(device)

    imgs = []
    for seed in seeds:
        latent_z = torch.from_numpy(np.random.RandomState(seed).randn(1, G_map.z_dim)).to(device)
        class_conditioning = one_hot(class_idx, num_classes=G_map.c_dim).to(device) if class_idx is not None else None
        latent_w_plus = G_map.forward(latent_z=latent_z, class_conditioning=class_conditioning, truncation=truncation)
        img = G_synth.forward(latent_w_plus=latent_w_plus, translation=translation, rotation=rotation)
        imgs.append(img)

    return torch.cat(imgs)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True, type=str, help="Path to .pkl file containing the model to use")
    parser.add_argument("--architecture", default="stylegan3", type=str, choices=["stylegan3"], help="The architecture of the model")
    parser.add_argument("--seeds", default="42", type=str, help="Comma separated list of seeds to generate images for. Use a dash to specify a range: e.g. '1,3,5,6-10'")
    parser.add_argument("--class_idx", default=None, type=int, help="Index of class to generate (only applicable to conditional models)")
    parser.add_argument("--truncation", default=1.0, type=float, help="Latent truncation value. Lower values give higher quality with less diversity; higer values vice versa")
    parser.add_argument("--translation", default="0,0", type=str, help="x,y offset values for StyleGAN3's input transformation matrix (translates image in latent space)")
    parser.add_argument("--rotation", default=0.0, type=float, help="Rotation value for StyleGAN3's input transformation matrix (rotates image in latent space)")
    parser.add_argument("--out_size", default="1024,1024", type=str, help="Desired width,height of output image: e.g. 1920,1080 or 720,1280")
    parser.add_argument("--resize_strategy", default="pad-zero", choices=["pad-zero", "stretch"], type=str, help="Strategy used to resize (in feature space) to achieve desired output resolution")
    parser.add_argument("--resize_layer", default=0, choices=list(range(15)), type=int, help="Which layer in the network to perform resizing at. Higher values are closer to resizing output pixels directly. Lower values have larger rounding increments (i.e. less flexible possible output sizes)")
    parser.add_argument("--out_dir", default="./workspace/", type=str, help="Directory to output images in")
    args = parser.parse_args()
    args.seeds = sum([([int(seed)] if not "-" in seed else list(range(int(seed.split("-")[0]), int(seed.split("-")[1])))) for seed in args.seeds.split(",")], [])
    args.truncation = torch.tensor([args.truncation])
    args.translation = torch.tensor([float(s) for s in args.translation.split(",")])
    args.rotation = torch.tensor(args.rotation)
    args.out_size = tuple(int(s) for s in args.out_size.split(","))
    # fmt: on

    imgs = generate_images(
        model_file=args.model_file,
        architecture=args.architecture,
        seeds=args.seeds,
        class_idx=args.class_idx,
        truncation=args.truncation,
        translation=args.translation,
        rotation=args.rotation,
        out_size=args.out_size,
        resize_strategy=args.resize_strategy,
        resize_layer=args.resize_layer,
    )

    for seed, img in zip(args.seeds, tensor2imgs(imgs)):
        img.save(
            f'{args.out_dir}/{Path(args.model_file.replace("/network-snapshot", "")).stem}_seed{seed}_{args.resize_strategy}.jpg'
        )
