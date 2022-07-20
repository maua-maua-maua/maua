import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import more_itertools
import torch
import transformers
from einops import rearrange
from PIL import Image
from torch.nn.functional import interpolate
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../submodules/ru_dalle")
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../submodules/VQGAN")
from rudalle import get_realesrgan, get_rudalle_model, get_tokenizer, get_vae, utils
from rudalle.dalle import MODELS
from rudalle.dalle.fp16 import FP16Module
from rudalle.dalle.image_attention import get_col_mask, get_conv_mask, get_row_mask
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


@torch.no_grad()
def oversample_generate_images(
    text,
    tokenizer,
    dalle,
    vae,
    top_k,
    top_p,
    num_images,
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
    real_w, real_h = (
        dalle.module.image_col_embeddings.weight.shape[0],
        dalle.module.image_row_embeddings.weight.shape[0],
    )

    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length).ravel().to(device)
    pil_images = []
    im_id = 0

    for chunk in more_itertools.chunked(range(num_images), bs):
        past_cache = None
        cache = None
        chunk_bs = len(chunk)

        attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
        grid = torch.zeros((h, w)).long().cuda()

        for idx in tqdm(range(input_ids.shape[0], total_seq_length - real_w * real_h + w * h)):
            idx -= text_seq_length

            y = idx // w
            x = idx % w
            x_from = max(0, min(w - real_w, x - real_w // 2))
            y_from = max(0, y - real_h // 2)
            samples = torch.cat((grid[y_from:y, x_from : x_from + real_w].flatten(), grid[y, x_from : x_from + x]))

            if past_cache:
                cache = list(map(list, cache.values()))
                for i, e in enumerate(cache):
                    for j, _ in enumerate(e):
                        t = cache[i][j]
                        t = t[..., :text_seq_length, :]
                        cache[i][j] = t
                cache = dict(zip(range(len(cache)), cache))
            past_cache = True

            inputs = torch.cat((input_ids, samples), dim=0).unsqueeze(0)
            logits, cache = dalle(inputs, attention_mask, cache=cache, use_cache=True, return_loss=False)
            logits = logits[:, -1, vocab_size:]
            logits /= temperature
            filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
            sample = (torch.empty_like(probs).uniform_().log() / probs).topk(1).indices  # faster than torch.multinomial
            grid[y, x] = sample

        codebooks = grid.reshape((1, -1))
        pil_images += utils.torch_tensors_to_pil_list(oversample_decode(vae, codebooks, h))
        if save_intermediate:
            for pi in range(-len(chunk), 0):
                if stretched_size:
                    pil_images[pi] = pil_images[pi].resize(stretched_size)
                pil_images[pi].save(f"{output_dir}/{model_name}_{im_id}.png")
                im_id += 1

    return pil_images


def _init_mask(text_tokens, w, h, is_bool_mask=False):
    attn_size = text_tokens + w * h
    mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.bool if is_bool_mask else torch.float32))
    return mask


def get_row_mask(text_tokens=256, w=32, h=32, is_bool_mask=False):
    mask = _init_mask(text_tokens, w, h, is_bool_mask=is_bool_mask)
    step = w + 1  # TODO w or h here?
    for col in range(text_tokens, mask.size(1)):
        mask[col + step :, col] = False if is_bool_mask else 0.0
    return mask


def get_col_mask(text_tokens=256, w=32, h=32, is_bool_mask=False):
    mask = _init_mask(text_tokens, w, h, is_bool_mask=is_bool_mask)
    step = h - 1  # TODO w or h here?
    for col in range(text_tokens, mask.size(1)):
        for i in range(1, mask.size(0), step + 1):
            mask[col + i : col + i + step, col] = False if is_bool_mask else 0.0
    return mask


def get_conv_mask(text_tokens=256, w=32, h=32, kernel=11, is_bool_mask=False, hf_version="v3"):
    mask = _init_mask(text_tokens, w, h, is_bool_mask=is_bool_mask)
    shift = kernel // 2
    for pos in range(text_tokens, mask.size(1)):
        mask[pos + 1 :, pos] = False if is_bool_mask else 0.0
        img = torch.zeros(h, w)
        pixel_id = pos - text_tokens
        row = pixel_id // w  # TODO w or h here?
        col = pixel_id % w  # TODO w or h here?
        for r in range(-shift, shift + 1):
            for c in range(-shift, shift + 1):
                if hf_version == "v2":
                    c_abs = (c + col) % w  # TODO w or h here?
                    r_abs = (r + row) % h  # TODO w or h here?
                elif hf_version == "v3":
                    c_abs = max(min(c + col, w - 1), 0)  # TODO w or h here?
                    r_abs = max(min(r + row, h - 1), 0)  # TODO w or h here?
                else:
                    raise ValueError(f"Unknown hf_version: {hf_version}")
                img[r_abs, c_abs] = 0.2
                cell_id = r_abs * w + c_abs  # TODO w or h here?
                if text_tokens + cell_id > pos:
                    mask[text_tokens + cell_id, pos] = True if is_bool_mask else 1.0
        img[row, col] = 1.0
    return mask


def get_image_pos_embeddings(image_input_ids, dalle, width, past_length=0):
    input_shape = image_input_ids.size()
    row_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=dalle.device) // width
    row_ids = row_ids.unsqueeze(0).view(-1, input_shape[-1])
    col_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=dalle.device) % width
    col_ids = col_ids.unsqueeze(0).view(-1, input_shape[-1])
    return dalle.image_row_embeddings(row_ids) + dalle.image_col_embeddings(col_ids)


@torch.no_grad()
def generate_images(
    text,
    tokenizer,
    dalle,
    vae,
    top_k,
    top_p,
    num_images,
    w=32,
    h=32,
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

    device = dalle.get_param("device")
    vocab_size = dalle.get_param("vocab_size")
    text_seq_length = dalle.get_param("text_seq_length")
    dalle.module.image_seq_length = image_seq_length = w * h
    dalle.module.total_seq_length = total_seq_length = text_seq_length + w * h
    if w != dalle.module.image_col_embeddings.weight.shape[0] or h != dalle.module.image_row_embeddings.weight.shape[0]:
        dalle.module.transformer.row_mask = get_row_mask(text_seq_length, w, h, is_bool_mask=True).to(device)
        dalle.module.transformer.col_mask = get_col_mask(text_seq_length, w, h, is_bool_mask=True).to(device)
        dalle.module.transformer.conv_mask = get_conv_mask(text_seq_length, w, h, is_bool_mask=True).to(device)
        dalle.module.image_row_embeddings.weight = torch.nn.Parameter(
            interpolate(dalle.module.image_row_embeddings.weight.T.unsqueeze(0), h).squeeze(0).T
        )
        dalle.module.image_col_embeddings.weight = torch.nn.Parameter(
            interpolate(dalle.module.image_col_embeddings.weight.T.unsqueeze(0), w).squeeze(0).T
        )
        dalle.module.get_image_pos_embeddings = partial(get_image_pos_embeddings, dalle=dalle.module, width=w)
        dalle.module.image_tokens_per_dim = -1  # unused if everything is set up correctly

    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)
    pil_images = []
    im_id = 0
    for chunk in more_itertools.chunked(range(num_images), bs):
        chunk_bs = len(chunk)
        attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
        out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)
        cache = None
        for idx in tqdm(range(out.shape[1], total_seq_length)):
            logits, cache = dalle(out, attention_mask, use_cache=True, cache=cache, return_loss=False)
            logits = logits[:, -1, vocab_size:]
            logits /= temperature
            filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)

        codebooks = out[:, -image_seq_length:]
        pil_images += utils.torch_tensors_to_pil_list(oversample_decode(vae, codebooks, h))
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
    top_p=0.99,
    top_k=2048,
    model_name="rudalle",
    output_dir="output/",
    save_intermediate=False,
    oversample=True,
):
    w, h = round(width / 8), round(height / 8)
    if oversample and (
        w != model.module.image_col_embeddings.weight.shape[0] or h != model.module.image_row_embeddings.weight.shape[0]
    ):
        return oversample_generate_images(
            input_text,
            tokenizer,
            model,
            vae,
            top_k=top_k,
            bs=batch_size,
            num_images=num_images,
            top_p=top_p,
            w=w,
            h=h,
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
            w=w,
            h=h,
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
    top_p=0.99,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    output_dir="output/",
    oversample=True,
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
        oversample (bool, optional): Whether to use the slower oversampling procedure or not. Works better with models that have not been finetuned for a given image size. Defaults to True.

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
        oversample=oversample,
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
    parser.add_argument("--top_p", type=float, default=0.99, help="Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to train on, using 'cpu' will take a long time!")
    parser.add_argument("--low_memory", action="store_true", help="Enable if you have less than 16 GB of (V)RAM to use gradient checkpointing (slower but more memory efficient)")
    parser.add_argument("--no_oversample", action="store_true", help="Disable oversampling procedure. Oversampling is slower but works better when sampling shapes different from what the model was trained on.")
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
        model = get_rudalle_model("Malevich", pretrained=False, fp16=True, device=device, cache_dir="modelzoo/")
        ckpt = torch.load(args.checkpoint)
        model.module.image_row_embeddings.weight = torch.nn.Parameter(
            torch.zeros_like(ckpt["image_row_embeddings.weight"])
        )
        model.module.image_col_embeddings.weight = torch.nn.Parameter(
            torch.zeros_like(ckpt["image_col_embeddings.weight"])
        )
        model.module.transformer.row_mask = torch.zeros_like(ckpt["transformer.row_mask"])
        model.module.transformer.col_mask = torch.zeros_like(ckpt["transformer.col_mask"])
        model.module.transformer.conv_mask = torch.zeros_like(ckpt["transformer.conv_mask"])
        w, h = model.module.image_col_embeddings.weight.shape[0], model.module.image_row_embeddings.weight.shape[0]
        model.module.image_seq_length = w * h
        model.module.total_seq_length = model.module.text_seq_length + model.module.image_seq_length
        model.module.load_state_dict(ckpt)
        model.module.get_image_pos_embeddings = partial(get_image_pos_embeddings, dalle=model.module, width=w)
        model.module.image_tokens_per_dim = -1  # unused if everything is set up correctly
        print(f"Loaded from {args.checkpoint}")

    output_name = args.output_name
    if output_name is None:
        output_name = str(uuid4())[:6]
        if args.input_text != "":
            output_name += "_" + args.input_text.replace(" ", "_")
        if args.checkpoint is not None:
            output_name += "_" + Path(args.checkpoint).stem
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
        oversample=not args.no_oversample,
    )
    for id, im in enumerate(outputs):
        im.save(f"{args.output_dir}/{output_name}_{id}.png")


if __name__ == "__main__":
    main(argument_parser().parse_args())
