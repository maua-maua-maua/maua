import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import more_itertools
import numpy as np
import torch
import transformers
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

sys.path.append(os.path.dirname(__file__) + "/../../submodules/ru_dalle")
sys.path.append(os.path.dirname(__file__) + "/../../submodules/VQGAN")
from rudalle import get_realesrgan, get_rudalle_model, get_tokenizer, get_vae, utils
from rudalle.dalle import MODELS
from rudalle.dalle.fp16 import FP16Module
from rudalle.dalle.model import DalleModel
from rudalle.pipelines import super_resolution

from . import SURREALIST_XL_DICT

MODELS.update({"Surrealist_XL": SURREALIST_XL_DICT})


def oversample_decode(self, img_seq, h):
    one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=self.num_tokens).float()
    z = one_hot_indices @ self.model.quantize.embed.weight
    z = rearrange(z, "b (h w) c -> b c h w", h=h)
    img = self.model.decode(z)
    img = (img.clamp(-1.0, 1.0) + 1) * 0.5
    return img


def oversample_generate_images(
    text,
    tokenizer,
    dalle,
    vae,
    top_k,
    top_p,
    num_images,
    image_prompts=None,
    temperature=1.0,
    bs=8,
    seed=None,
    w=32,
    h=32,
    stretched_size=None,
    model_name="rudalle",
    output_dir="output/",
    save_intermediate=False,
):
    if seed is not None:
        utils.seed_everything(seed)
    vocab_size = dalle.get_param("vocab_size")
    text_seq_length = dalle.get_param("text_seq_length")
    total_seq_length = dalle.get_param("total_seq_length")
    device = dalle.get_param("device")
    real = 32

    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)
    pil_images = []
    cache = None
    past_cache = None
    im_id = 0

    for chunk in more_itertools.chunked(range(num_images), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
            out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)
            grid = torch.zeros((h, w)).long().cuda()
            sample_scores = []
            if image_prompts is not None:
                prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                prompts = prompts.repeat(chunk_bs, 1)
            for idx in tqdm(range(out.shape[1], total_seq_length - real * real + w * h)):
                idx -= text_seq_length
                if image_prompts is not None and idx in prompts_idx:
                    out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                else:
                    y = idx // w
                    x = idx % w
                    x_from = max(0, min(w - real, x - real // 2))
                    y_from = max(0, y - real // 2)
                    outs = []
                    xs = []
                    for row in range(y_from, y):
                        for col in range(x_from, x_from + real):
                            outs.append(grid[row, col].item())
                            xs.append((row, col))
                    for col in range(x_from, x):
                        outs.append(grid[y, col].item())
                        xs.append((y, col))
                    if past_cache is not None:
                        cache = list(map(list, cache.values()))
                        for i, e in enumerate(cache):
                            for j, _ in enumerate(e):
                                t = cache[i][j]
                                t = t[..., :text_seq_length, :]
                                cache[i][j] = t
                        cache = dict(zip(range(len(cache)), cache))
                    past_cache = xs
                    logits, cache = dalle(
                        torch.cat(
                            (input_ids.to(device).ravel(), torch.from_numpy(np.asarray(outs)).long().to(device)), dim=0
                        ).unsqueeze(0),
                        attention_mask,
                        cache=cache,
                        use_cache=True,
                        return_loss=False,
                    )
                    logits = logits[:, :, vocab_size:].view((-1, logits.shape[-1] - vocab_size))
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    sample_scores.append(probs[torch.arange(probs.size(0)), sample.transpose(0, 1)])
                    sample, xs = sample[-1:], xs[-1:]
                    grid[y, x] = sample.item()
            codebooks = grid.reshape((1, -1))
            pil_images += utils.torch_tensors_to_pil_list(oversample_decode(vae, codebooks, h))
            if save_intermediate:
                for pi in range(-len(chunk), 0):
                    if stretched_size:
                        pil_images[pi] = pil_images[pi].resize(stretched_size)
                    pil_images[pi].save(f"{output_dir}/{model_name}_{im_id}.png")
                    im_id += 1

    return pil_images


def generate_images(
    text,
    tokenizer,
    dalle,
    vae,
    top_k,
    top_p,
    num_images,
    image_prompts=None,
    temperature=1.0,
    bs=8,
    seed=None,
    stretched_size=None,
    model_name="rudalle",
    output_dir="output/",
    save_intermediate=False,
):
    if seed is not None:
        utils.seed_everything(seed)
    vocab_size = dalle.get_param("vocab_size")
    text_seq_length = dalle.get_param("text_seq_length")
    image_seq_length = dalle.get_param("image_seq_length")
    total_seq_length = dalle.get_param("total_seq_length")
    device = dalle.get_param("device")

    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)
    pil_images = []
    im_id = 0
    for chunk in more_itertools.chunked(range(num_images), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
            out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)
            cache = None
            if image_prompts is not None:
                prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                prompts = prompts.repeat(chunk_bs, 1)
            for idx in tqdm(range(out.shape[1], total_seq_length)):
                idx -= text_seq_length
                if image_prompts is not None and idx in prompts_idx:
                    out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                else:
                    logits, cache = dalle(out, attention_mask, use_cache=True, cache=cache, return_loss=False)
                    logits = logits[:, -1, vocab_size:]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    out = torch.cat((out, sample), dim=-1)
            codebooks = out[:, -image_seq_length:]
            pil_images += utils.torch_tensors_to_pil_list(vae.decode(codebooks))
            if save_intermediate:
                for pi in range(-len(chunk), 0):
                    if stretched_size:
                        pil_images[pi] = pil_images[pi].resize(stretched_size)
                    pil_images[pi].save(f"{output_dir}/{model_name}_{im_id}.png")
                    im_id += 1
    return pil_images


def sample_images(
    model,
    vae,
    tokenizer,
    input_text,
    batch_size,
    num_images,
    height=256,
    width=256,
    stretched_size=None,
    top_p=0.999,
    top_k=2048,
    model_name="rudalle",
    output_dir="output/",
    save_intermediate=False,
):
    if width != 256 or height != 256:
        return oversample_generate_images(
            input_text,
            tokenizer,
            model,
            vae,
            top_k=top_k,
            bs=batch_size,
            num_images=num_images,
            top_p=top_p,
            w=round(width / 8),
            h=round(height / 8),
            stretched_size=stretched_size,
            model_name=model_name,
            output_dir=output_dir,
            save_intermediate=save_intermediate,
        )
    else:
        return generate_images(
            input_text,
            tokenizer,
            model,
            vae,
            top_k=top_k,
            bs=batch_size,
            num_images=num_images,
            top_p=top_p,
            stretched_size=stretched_size,
            model_name=model_name,
            output_dir=output_dir,
            save_intermediate=save_intermediate,
        )


@torch.inference_mode()
def generate(
    model: Union[FP16Module, DalleModel],
    model_name="rudalle",
    input_text="",
    num_outputs=8,
    batch_size=4,
    height=256,
    width=256,
    stretched_size: Optional[Tuple[int, int]] = None,
    upscale=1,
    top_p=0.999,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    output_dir="output/",
    save_intermediate=False,
) -> List[Image.Image]:
    """Generate images by sampling from the RuDALL-E model.

    Args:
        model (Union[FP16Module, DalleModel]): RuDALL-E model to generate images with.
        model_name (str, optional): Name which images will be saved with. Defaults to "rudalle".
        input_text (str, optional): Text to generate image with (English). Defaults to "".
        num_outputs (int, optional): Number of images to generate. Defaults to 8.
        batch_size (int, optional): Number of images to sample at once. Higher batch size requires more memory, but will be faster per sample overall. Defaults to 4.
        height (int, optional): Height for output images. If set to more than 256, generation will be significantly slower and the model will sample more tokens than it is trained for. This can lead to unexpected results. Defaults to 256.
        width (int, optional): Width for output images. If set to more than 256, generation will be significantly slower and the model will sample more tokens than it is trained for. This can lead to unexpected results. Defaults to 256.
        stretched_size (Optional[Tuple[int, int]], optional): If RuDALL-E model is trained on stretched images, specify the size to re-stretch sampled images to. Defaults to None.
        upscale (int, optional): Factor for RealESRGAN to upscale outputs to. Defaults to 1.
        top_p (float, optional): Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999. Defaults to 0.999.
        device (torch.device, optional): Device to sample on. Defaults to 'cuda:0' if CUDA is available, otherwise 'cpu'.
        output_dir (str, optional): Directory to save output images in. Defaults to "output/".
        save_intermediate (bool, optional): Whether to save intermediate results immediately on completing samples. Defaults to False.

    Returns:
        List[PIL.Image]: num_ouputs PIL.Images sampled from the model
    """
    mname = "Helsinki-NLP/opus-mt-en-ru"
    translate_tokenizer = MarianTokenizer.from_pretrained(mname)
    translator = AutoModelForSeq2SeqLM.from_pretrained(mname)

    input_ids = translate_tokenizer.encode(input_text, return_tensors="pt")
    outputs = translator.generate(input_ids)
    text = translate_tokenizer.decode(outputs[0], skip_special_tokens=True)

    vae = get_vae(cache_dir="modelzoo/").to(device)

    if width != 256 or height != 256:
        vae.oversample_decode = partial(oversample_decode, vae)

    pil_images = sample_images(
        model=model,
        vae=vae,
        tokenizer=get_tokenizer(),
        input_text=text,
        num_images=num_outputs,
        batch_size=batch_size,
        height=height,
        width=width,
        stretched_size=stretched_size,
        top_p=top_p,
        model_name=model_name,
        output_dir=output_dir,
        save_intermediate=save_intermediate,
    )
    if upscale > 1:
        pil_images = super_resolution(
            pil_images, get_realesrgan(f"x{upscale}", device=device, fp16=True, cache_dir="modelzoo/"), batch_size=1
        )
    return pil_images


def argument_parser():
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str, default="", help="Input text to sample images for after training. Will only have effect if you train with low number of steps / low learning rate or train with captions.")
    parser.add_argument("--num_outputs", type=int, default=8, help="Number of images to generate after finetuning")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to sample at once. Higher batch size requires more memory, but will be faster per sample overall. Inference batches can be bigger as we don't need to store gradients for training.")
    parser.add_argument("--size", type=str, default='256,256', help="width,height of images to sample")
    parser.add_argument("--stretch_size", type=str, default=None, help="width,height to stretch sampled images to (will only give decent results if model was finetuned with this stretch size).")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2, 4, 8], help="Use RealESRGAN to upscale outputs.")
    parser.add_argument("--top_p", type=float, default=0.999, help="Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to train on, using 'cpu' will take a long time!")
    parser.add_argument("--low_memory", action="store_true", help="Enable if you have less than 16 GB of (V)RAM to use gradient checkpointing (slower but more memory efficient)")
    parser.add_argument("--checkpoint", type=str, default=None, help=f"Checkpoint to resume from. Either a path to a trained RuDALL-E checkpoint or one of {list(MODELS.keys())}.")
    parser.add_argument("--output_name", type=str, default=None, help="Name to save images under.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save output images in.")
    # fmt: on
    return parser


def main(args):
    device = torch.device(args.device)

    width, height = [int(v) for v in args.size.split(",")]

    if args.stretch_size is not None:
        stretched_size = tuple(int(v) for v in args.stretch_size.split(","))
    else:
        stretched_size = None

    if args.checkpoint is None:
        args.checkpoint = "Malevich"
    if args.checkpoint in list(MODELS.keys()):
        model = get_rudalle_model(args.checkpoint, pretrained=True, fp16=True, device=device, cache_dir="modelzoo/")
    else:
        model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=device, cache_dir="modelzoo/")
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded from {args.checkpoint}")

    output_name = args.output_name
    if output_name is None:
        if args.input_text != "":
            output_name = args.input_text.replace(" ", "_")
        if args.checkpoint is not None:
            output_name = Path(args.checkpoint).stem
        else:
            output_name = ""
    output_name += "_rudalle"

    outputs = generate(
        model,
        model_name=output_name,
        input_text=args.input_text,
        num_outputs=args.num_outputs,
        batch_size=args.batch_size,
        height=height,
        width=width,
        stretched_size=stretched_size,
        upscale=args.upscale,
        top_p=args.top_p,
        device=args.device,
        output_dir=args.output_dir,
        save_intermediate=True,
    )
    for id, im in enumerate(outputs):
        im.save(f"{args.output_dir}/{output_name}_{id}.png")


if __name__ == "__main__":
    main(argument_parser().parse_args())
