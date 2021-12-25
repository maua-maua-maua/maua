import argparse
import gc
import os
import random
import sys
from functools import partial
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

import more_itertools
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import transformers
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, AutoModelForSeq2SeqLM, MarianTokenizer

sys.path.append("submodules/ru-dalle")
sys.path.append("submodules/VQGAN")
from rudalle import get_realesrgan, get_rudalle_model, get_tokenizer, get_vae, utils
from rudalle.dalle.fp16 import FP16Module
from rudalle.dalle.model import DalleModel
from rudalle.dalle.utils import exists, is_empty
from rudalle.pipelines import super_resolution


def infiniter(dataloader):
    while True:
        for sample in dataloader:
            yield sample


class RuDalleDataset(Dataset):
    def __init__(self, images, captions, height, width, stretch, vae, tokenizer, text_seq_length, device):
        self.text_seq_length = text_seq_length
        self.tokenizer = tokenizer
        self.device = device

        self.image_transform = T.Compose(
            [
                (
                    T.Resize((height, width), antialias=True)
                    if stretch
                    else T.Compose([T.Resize(max(height, width), antialias=True), T.CenterCrop((height, width))])
                ),
                T.ToTensor(),
            ]
        )

        mname = "Helsinki-NLP/opus-mt-en-ru"
        translate_tokenizer = MarianTokenizer.from_pretrained(mname)
        translator = AutoModelForSeq2SeqLM.from_pretrained(mname)

        self.samples = list(
            zip(
                [
                    vae.get_codebook_indices(
                        self.image_transform(
                            Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image
                        )
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    .cpu()
                    .squeeze(0)
                    for image in tqdm(images, desc="preprocessing images...")
                ],
                [
                    self.tokenizer.encode_text(
                        translate_tokenizer.decode(
                            translator.generate(translate_tokenizer.encode(text, return_tensors="pt"))[0],
                            skip_special_tokens=True,
                        ),
                        text_seq_length=self.text_seq_length,
                    ).squeeze(0)
                    for text in tqdm(captions, desc="preprocessing captions...")
                ]
                if len(captions) != 0
                else [self.tokenizer.encode_text("", text_seq_length=self.text_seq_length).squeeze(0)] * len(images),
            )
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        image_tokens, text_tokens = self.samples[item]
        return image_tokens.to(self.device), text_tokens.to(self.device)


def freeze(
    model,
    freeze_emb=True,
    freeze_ln=False,
    freeze_attn=False,
    freeze_ff=True,
    freeze_other=True,
):
    for name, p in model.module.named_parameters():
        name = name.lower()
        if "ln" in name or "norm" in name:
            p.requires_grad = not freeze_ln
        elif "embeddings" in name:
            p.requires_grad = not freeze_emb
        elif "mlp" in name:
            p.requires_grad = not freeze_ff
        elif "attn" in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
    return model


def train(
    model,
    dataset,
    model_name,
    lr=1e-4,
    steps=100,
    batch_size=1,
    train_text=False,
    gradient_checkpointing=False,
    save_dir="modelzoo/",
):
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    model = freeze(model=model, freeze_emb=False, freeze_ln=False, freeze_attn=True, freeze_ff=True, freeze_other=False)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, final_div_factor=500, steps_per_epoch=int(np.ceil(steps / 30)), epochs=30
    )

    total_seq_length = model.get_param("total_seq_length")

    try:
        step = 0
        with tqdm(total=steps, desc="finetuning...") as progress:
            for text_tokens, image_tokens in infiniter(train_dataloader):
                optimizer.zero_grad()

                input_ids = torch.cat((text_tokens, image_tokens), dim=1)
                attention_mask = torch.tril(
                    torch.ones((batch_size, 1, total_seq_length, total_seq_length), device=model.get_param("device"))
                )

                _, loss = forward(
                    model.module,
                    input_ids,
                    attention_mask.half(),
                    return_loss=True,
                    use_cache=False,
                    gradient_checkpointing=gradient_checkpointing,
                )
                loss = loss["image"]
                if train_text:
                    loss += loss["text"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.24)

                optimizer.step()
                scheduler.step()
                step += 1
                progress.update()
                progress.set_postfix({"loss": loss.item()})

                if step >= steps:
                    break

        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_last.pt"))

    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_keyboardinterrupt.pt"))
        exit(0)
    except Exception as err:
        raise err


class Layer(torch.nn.Module):
    def __init__(self, x, f, *args, **kwargs):
        super(Layer, self).__init__()
        self.x = x
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.f(self.x(x, *self.args, **self.kwargs))


def forward(self, input_ids, attention_mask, return_loss=False, use_cache=False, gradient_checkpointing=None):
    text = input_ids[:, : self.text_seq_length]
    text_range = torch.arange(self.text_seq_length)
    text_range += self.vocab_size - self.text_seq_length
    text_range = text_range.to(self.device)
    text = torch.where(text == 0, text_range, text)

    text = F.pad(text, (1, 0), value=2)
    text_embeddings = self.text_embeddings(text) + self.text_pos_embeddings(
        torch.arange(text.shape[1], device=self.device)
    )

    image_input_ids = input_ids[:, self.text_seq_length :]

    if exists(image_input_ids) and not is_empty(image_input_ids):
        image_embeddings = self.image_embeddings(image_input_ids) + self.get_image_pos_embeddings(
            image_input_ids, past_length=0
        )
        embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
    else:
        embeddings = text_embeddings

    if embeddings.shape[1] > self.total_seq_length:
        embeddings = embeddings[:, :-1]

    alpha = 0.1
    embeddings = embeddings * alpha + embeddings.detach() * (1 - alpha)

    attention_mask = attention_mask[:, :, : embeddings.shape[1], : embeddings.shape[1]]
    t = self.transformer
    layers = []
    layernorms = []
    if not layernorms:
        norm_every = 0
    else:
        norm_every = len(t.layers) // len(layernorms)
    for i in range(len(t.layers)):
        layers.append(
            Layer(
                t.layers[i],
                lambda x: x[0] * layernorms[i // norm_every][0] + layernorms[i // norm_every][1]
                if norm_every and i % norm_every == 0
                else x[0],
                torch.mul(
                    attention_mask,
                    t._get_layer_mask(i)[
                        : attention_mask.size(2),
                        : attention_mask.size(3),
                    ],
                ),
                use_cache=False,
            )
        )
    if gradient_checkpointing:
        embeddings = torch.utils.checkpoint.checkpoint_sequential(layers, 6, embeddings)
        transformer_output = embeddings
        present_has_cache = False
    else:
        hidden_states = embeddings
        for i in range(len(t.layers)):
            mask = torch.mul(attention_mask, t._get_layer_mask(i)[: attention_mask.size(2), : attention_mask.size(3)])
            hidden_states, present_has_cache = t.layers[i](hidden_states, mask, use_cache=use_cache)
        transformer_output = hidden_states
    transformer_output = self.transformer.final_layernorm(transformer_output)

    logits = self.to_logits(transformer_output)
    if return_loss is False:
        return logits, present_has_cache

    labels = torch.cat((text[:, 1:], image_input_ids), dim=1).contiguous().long()
    logits = rearrange(logits, "b n c -> b c n")

    text_logits = logits[:, : self.vocab_size, : self.text_seq_length].contiguous().float()
    image_logits = logits[:, self.vocab_size :, self.text_seq_length :].contiguous().float()

    loss_text = F.cross_entropy(text_logits, labels[:, : self.text_seq_length])
    loss_img = F.cross_entropy(image_logits, labels[:, self.text_seq_length :])

    loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
    return loss, {"text": loss_text.data.detach().float(), "image": loss_img}


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
    try:
        for chunk in more_itertools.chunked(range(num_images), bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = torch.tril(
                    torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device)
                )
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
                                (input_ids.to(device).ravel(), torch.from_numpy(np.asarray(outs)).long().to(device)),
                                dim=0,
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

    except Exception as e:
        print(e)
        pass
    except KeyboardInterrupt:
        pass
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
    use_cache=True,
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
                    logits, cache = dalle(out, attention_mask, use_cache=use_cache, cache=cache, return_loss=False)
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


def finetune(
    images: List[Union[str, Path, Image.Image]],
    captions: List[str] = [],
    model_name="rudalle",
    steps=100,
    lr=1e-4,
    batch_size=1,
    height=256,
    width=256,
    stretch=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    low_memory=False,
    checkpoint: Optional[Union[str, Path]] = None,
    save_dir="modelzoo/",
) -> Tuple[Union[FP16Module, DalleModel], Optional[Tuple[int, int]]]:
    """Finetune a RuDALL-E model on a set of images (and possibly captions).

    Args:
        images (List[Union[str, Path, Image.Image]]): List of PIL.Images or image files to finetune on
        captions (List[str], optional): A list of captions for each image. Defaults to [].
        model_name (str, optional): Model name to save finetuned model with. Defaults to "rudalle".
        steps (int, optional): Number of batches to finetune for. More steps will converge to outputs that are closer to the inputs (which also means less variation). Defaults to 100.
        lr ([type], optional): Starting learning rate (decays by a factor of 500 over training). A high learning rate will converge faster (which also means less variation). 1e-4 to 1e-5 is a good starting range (with 1e-5 resulting in more varied outputs). Defaults to 1e-4.
        batch_size (int, optional): Number of images for each training step. Higher batch size requires more memory, but will be faster per sample overall. Defaults to 1.
        height (int, optional): Height for output images. If set to more than 256, generation will be significantly slower and the model will sample more tokens than it is trained for. This can lead to unexpected results. Defaults to 256.
        width (int, optional): Width for output images. If set to more than 256, generation will be significantly slower and the model will sample more tokens than it is trained for. This can lead to unexpected results. Defaults to 256.
        stretch (bool, optional): Squash images down to fixed size for training and stretch back to original size after sampling. This can significantly improve training/sampling time while still yielding good results. All training images should have the same aspect ratio. Defaults to False.
        device (torch.device, optional): The device to train on, using 'cpu' will take a long time!. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        low_memory (bool, optional): Enable if you have less than 16 GB of (V)RAM to use gradient checkpointing (slower but more memory efficient). Defaults to False.
        checkpoint (Optional[Union[str, Path]], optional): Checkpoint to resume from. Defaults to official Malevich checkpoint. Defaults to None.
        save_dir (str, optional): Directory to save finetuned checkpoints in. Defaults to "modelzoo/".

    Returns:
        Tuple[Union[FP16Module, DalleModel], Optional[Tuple[int, int]]]: The finetuned model and the size that sampled images should be stretched to if `stretch` is enabled.
    """

    assert len(captions) == 0 or len(captions) == len(
        images
    ), "When specifying captions, the number of images must match exactly."

    low_mem = low_memory
    if width != 256 or height != 256:
        low_mem = True

    stretched_size = None
    if stretch:
        original_size = (Image.open(images[0]) if isinstance(images[0], (str, Path)) else images[0]).size
        w, h = original_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = min(w, h, 256)
        new_short, new_long = requested_new_short, int(requested_new_short * long / short)
        stretched_size = (new_short, new_long) if w <= h else (new_long, new_short)

    model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=device, cache_dir="modelzoo/")
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        print(f"Loaded from {checkpoint}")

    tokenizer = get_tokenizer()

    train(
        model,
        RuDalleDataset(
            images=images,
            captions=captions,
            height=height,
            width=width,
            stretch=stretch,
            vae=get_vae().to(device),
            tokenizer=tokenizer,
            text_seq_length=model.get_param("text_seq_length"),
            device=device,
        ),
        model_name,
        lr=lr,
        steps=steps,
        batch_size=batch_size,
        train_text=len(captions) > 0,
        gradient_checkpointing=low_mem,
        save_dir=save_dir,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return model, stretched_size


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


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with images to train on")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of example images from input_dir to train on (None => All images will be used)")
    parser.add_argument("--input_imgs", type=list, default=[], nargs="*", help="List of images to train on")
    parser.add_argument("--captions", type=list, default=[], nargs="*", help="List of a caption for each image")
    parser.add_argument("--input_text", type=str, default="", help="Input text to sample images for after training. Will only have effect if you train with low number of steps / low learning rate or train with captions.")
    parser.add_argument("--num_outputs", type=int, default=8, help="Number of images to generate after finetuning")
    parser.add_argument("--steps", type=int, default=100, help="Number of batches to finetune for. More steps will converge to outputs that are closer to the inputs (which also means less variation).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Starting learning rate (decays by a factor of 500 over training). A high learning rate will converge faster (which also means less variation). 1e-4 to 1e-5 is a good starting range (with 1e-5 resulting in more varied outputs).")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of images for each training step. Higher batch size requires more memory, but will be faster per sample overall.")
    parser.add_argument("--inference_batch_size", type=int, default=1, help="Number of images to sample at once. Higher batch size requires more memory, but will be faster per sample overall. Inference batches can be bigger as we don't need to store gradients for training.")
    parser.add_argument("--size", type=str, default='256,256', help="width,height of images to generate")
    parser.add_argument("--stretch", action="store_true", help="Squash images down to fixed size for training and stretch back to original size after sampling. This can significantly improve training/sampling time while still yielding good results. All training images should have the same aspect ratio.")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2, 4, 8], help="Use RealESRGAN to upscale outputs.")
    parser.add_argument("--top_p", type=float, default=0.999, help="Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to train on, using 'cpu' will take a long time!")
    parser.add_argument("--low_memory", action="store_true", help="Enable if you have less than 16 GB of (V)RAM to use gradient checkpointing (slower but more memory efficient)")
    parser.add_argument("--sample_only", action="store_true", help="Skip finetuning and just sample images from a pre-trained model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from. Defaults to official Malevich checkpoint.")
    parser.add_argument("--save_dir", type=str, default="modelzoo/", help="Directory to save finetuned checkpoints in.")
    parser.add_argument("--model_name", type=str, default=None, help="Name for finetuned checkpoints. Will default to the name of input_dir or the first input_img.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save output images in.")
    args = parser.parse_args()
    # fmt: on

    assert len(args.captions) == 0 or len(args.input_imgs) == len(
        args.captions
    ), "When specifying captions, the number of input_imgs must match exactly."
    assert (
        len(args.input_imgs) > 0 or args.input_dir is not None
    ), "You must specify either a list of input_imgs or an input_dir to train with."

    device = torch.device(args.device)

    images = args.input_imgs
    if len(images) == 0:
        images = glob(args.input_dir + "/*", recursive=True)
        if args.num_examples is not None:
            images = random.choices(images, k=args.num_examples)

    model_name = f"rudalle_finetuned_{Path(args.input_dir).stem if args.input_dir is not None else Path(args.input_imgs[0]).stem}"
    width, height = [int(v) for v in args.size.split(",")]

    if not args.sample_only:
        model, stretched_size = finetune(
            images,
            captions=args.captions,
            model_name=model_name,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.train_batch_size,
            height=height,
            width=width,
            stretch=args.stretch,
            device=args.device,
            low_memory=args.low_memory,
            checkpoint=args.checkpoint,
            save_dir=args.save_dir,
        )
    else:
        # if sample_only, we need to load in the correct model and calcualate stretched size ourself

        model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=device, cache_dir="modelzoo/")
        if args.checkpoint is not None:
            model.load_state_dict(torch.load(args.checkpoint))
            print(f"Loaded from {args.checkpoint}")

        stretched_size = None
        if args.stretch:
            original_size = (Image.open(images[0]) if isinstance(images[0], (str, Path)) else images[0]).size
            w, h = original_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = min(w, h, 256)
            new_short, new_long = requested_new_short, int(requested_new_short * long / short)
            stretched_size = (new_short, new_long) if w <= h else (new_long, new_short)

    outputs = generate(
        model,
        model_name=model_name,
        input_text=args.input_text,
        num_outputs=args.num_outputs,
        batch_size=args.inference_batch_size,
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
        im.save(f"{args.output_dir}/{model_name}_{id}.png")
