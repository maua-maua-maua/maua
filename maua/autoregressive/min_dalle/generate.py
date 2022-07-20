import os
import sys

import clip
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../submodules/minDALLE")
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
