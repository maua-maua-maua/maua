import os
import sys

import clip
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(__file__) + "/../../submodules/minDALLE")
from dalle.models import Dalle
from dalle.utils.utils import clip_score


def generate(prompt, num_candidates, top_k, top_p, device):
    model = Dalle.from_pretrained("minDALL-E/1.3B")
    model.to(device=device)

    images = (
        model.sampling(
            prompt=prompt,
            top_k=top_k,
            top_p=top_p,
            softmax_temperature=1.0,
            num_candidates=num_candidates,
            device=device,
        )
        .cpu()
        .numpy()
    )
    images = np.transpose(images, (0, 2, 3, 1))

    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
    model_clip.to(device=device)
    rank = clip_score(
        prompt=prompt, images=images, model_clip=model_clip, preprocess_clip=preprocess_clip, device=device
    )

    return images[rank]


def argument_parser():
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Input text to sample images.")
    parser.add_argument("--num_candidates", type=int, default=32, help="Number of images to generate in total")
    parser.add_argument("--num_outputs", type=int, default=8, help="Number of images to output based on best CLIP scores")
    parser.add_argument("--top_k", type=float, default=256, help="Should probably be set no higher than 256.")
    parser.add_argument("--top_p", type=float, default=None, help="Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to train on, using 'cpu' will take a long time!")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save output images in.")
    # fmt: on
    return parser


def main(args):
    images = generate(
        prompt=args.prompt,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        top_p=args.top_p,
        device=torch.device(args.device),
    )

    output_name = args.prompt.replace(" ", "_") + "_mindalle"
    for id, im in enumerate((images[: args.num_outputs] * 255).astype(np.uint8)):
        Image.fromarray(im).save(f"{args.output_dir}/{output_name}_{id}.png")


if __name__ == "__main__":
    main(argument_parser().parse_args())
