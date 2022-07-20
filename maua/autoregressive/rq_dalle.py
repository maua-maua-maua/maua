# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from glob import glob
from pathlib import Path
from shutil import rmtree

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from ..utility import download, unzip

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../submodules/rq_vae_transformer/")
from rqvae.models import create_model
from rqvae.txtimg_datasets.tokenizers import create_tokenizer
from rqvae.utils.config import augment_arch_defaults, load_config

URLS = {
    "ffhq": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/d47570aeff6ba300735606a806f54663/ffhq.tar.gz",
    "church": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/deeb3e0ac6e09923754e3e594ede7b01/church.tar.gz",
    "cat": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/92b4e6a9ace09c9ab8ff9d3b3e688367/cat.tar.gz",
    "bedroom": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/06b72c164cd2fe64fc8ebd6b42b0040f/bedroom.tar.gz",
    "imagenet_480M": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/7518a004fe39120fcffbba76005dc6c3/imagenet_480M.tar.gz",
    "imagenet_821M": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/dcd39292319104da5577dec3956bfdcc/imagenet_821M.tar.gz",
    "imagenet_1.4B": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/f5cf4e5f3f0b5088d52cbb5e85c1077f/imagenet_1.4B.tar.gz",
    "imagenet_1.4B_rqvae_50e": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/6714b47bb9382076923590eff08b1ee5/imagenet_1.4B_rqvae_50e.tar.gz",
    "imagenet_3.8B_rqvae_50e": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/e1ee2fef2928f7fd31f53a8348f08b88/imagenet_3.8B_rqvae_50e.tar.gz",
    "cc3m": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/dcd95e8f08408e113aab6451fae895f5/cc3m.tar.gz",
    "cc3m_cc12m_yfcc": "https://arena.kakaocdn.net/brainrepo/models/RQVAE/3a8429cd7ec0e0f2b66fca94804c79d5/cc3m_cc12m_yfcc.tar.gz",
}


class TextEncoder:
    def __init__(self, tokenizer_name, context_length=64, lowercase=True):
        self.tokenizer = create_tokenizer(tokenizer_name, lowercase=lowercase)
        self.context_length = context_length

        self.tokenizer.add_special_tokens(["[PAD]"])
        self.tokenizer.enable_padding(length=self.context_length, pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.tokenizer.enable_truncation(max_length=self.context_length)

    def encode(self, texts):
        output = self.tokenizer.encode(texts)
        ids = output.ids

        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)

        return ids

    def __call__(self, texts):
        return self.encode(texts)


@torch.inference_mode()
def load_model(path, ema=False, map_location="cpu"):
    model_config = os.path.join(os.path.dirname(path), "config.yaml")
    config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)

    model, _ = create_model(config.arch, ema=False)
    if ema:
        ckpt = torch.load(path, map_location=map_location)["state_dict_ema"]
    else:
        ckpt = torch.load(path, map_location=map_location)["state_dict"]
    model.load_state_dict(ckpt)

    return model, config


@torch.inference_mode()
def get_initial_sample(batch_sample_shape, device=torch.device("cuda")):
    partial_sample = torch.zeros(*batch_sample_shape, dtype=torch.long, device=device)
    return partial_sample


@torch.inference_mode()
def get_clip_score(pixels, texts, model_clip, preprocess_clip, device=torch.device("cuda")):
    pixels = pixels.cpu().numpy()
    pixels = np.transpose(pixels, (0, 2, 3, 1))

    images = [preprocess_clip(Image.fromarray((pixel * 255).astype(np.uint8))) for pixel in pixels]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(texts).to(device=device)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()

    return scores


@torch.inference_mode()
def get_generated_images_by_texts(
    model_ar,
    model_vqvae,
    text_encoder,
    model_clip,
    preprocess_clip,
    text_prompts,
    num_samples,
    temperature,
    top_k,
    top_p,
    amp=True,
    fast=True,
    is_tqdm=True,
):

    sample_shape = model_ar.get_block_size()

    text_cond = text_encoder(text_prompts).unsqueeze(0).repeat(num_samples, 1).cuda()

    initial_codes = get_initial_sample([num_samples, *sample_shape])
    generated_codes = model_ar.sample(
        initial_codes,
        model_vqvae,
        cond=text_cond,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        amp=amp,
        fast=fast,
        is_tqdm=is_tqdm,
    )
    pixels = torch.cat(
        [model_vqvae.decode_code(generated_codes[i : i + 1]) for i in range(generated_codes.size(0))], dim=0
    )

    clip_scores = get_clip_score(pixels, text_prompts, model_clip, preprocess_clip)

    reranked_idxs = clip_scores.argsort(descending=True)
    reranked_pixels = pixels[reranked_idxs]

    return reranked_pixels


@torch.inference_mode()
def main(
    text_prompts,
    num_samples=64,
    sampling_ratio=0.1,
    batch_size=16,
    temperature=1.0,
    top_k=1024,
    top_p=0.95,
    checkpoint_dir="modelzoo/rqvae_cc3m_cc12m_yfcc/",
    clip_model="ViT-B/32",
    make_grid=False,
    out_dir="output/",
):
    if not os.path.exists(checkpoint_dir):
        try:
            path = download(URLS[Path(checkpoint_dir).stem.replace("rqvae_")], checkpoint_dir)
            file = glob(path + "*.tar.gz")[0]
            unzip(file, checkpoint_dir)
            rmtree(file)
        except:
            raise Exception("Checkpoint not found!")
    model_vqvae, _ = load_model(f"{checkpoint_dir}/stage1/model.pt")
    model_ar, config = load_model(f"{checkpoint_dir}/stage2/model.pt", ema=False, map_location="cuda")
    model_ar = model_ar.cuda().eval()
    model_vqvae = model_vqvae.cuda().eval()
    model_clip, preprocess_clip = clip.load(clip_model, device="cuda")
    model_clip = model_clip.cuda().eval()
    text_encoder = TextEncoder(tokenizer_name=config.dataset.txt_tok_name, context_length=config.dataset.context_length)

    images = []
    for b in tqdm(range(0, round(num_samples / sampling_ratio), batch_size)):
        pixels = get_generated_images_by_texts(
            model_ar=model_ar,
            model_vqvae=model_vqvae,
            text_encoder=text_encoder,
            model_clip=model_clip,
            preprocess_clip=preprocess_clip,
            text_prompts=text_prompts,
            num_samples=batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            amp=False,
            is_tqdm=False,
        )
        images.append(pixels.cpu())
    images = torch.cat(images)
    clip_scores = get_clip_score(images, text_prompts, model_clip, preprocess_clip)
    reranked_idxs = clip_scores.argsort(descending=True)
    images = images[reranked_idxs[:num_samples]]
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)

    if make_grid:
        grid = torchvision.utils.make_grid(images, nrow=round(np.sqrt(len(images)) * 4 / 3))
        img = Image.fromarray(np.uint8(grid.numpy().transpose([1, 2, 0]) * 255))
        img.save(f"{out_dir}/{text_prompts}_temp_{temperature}_top_k_{top_k}_top_p_{top_p}.jpg")
    else:
        for i, img in enumerate(images):
            img = Image.fromarray(np.uint8(img.numpy().transpose([1, 2, 0]) * 255))
            img.save(f"{out_dir}/{text_prompts}_{i}_temp_{temperature}_top_k_{top_k}_top_p_{top_p}.jpg")
