import gc
import os
import random
import sys
from functools import partial
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from bitsandbytes.optim import Adam8bit
from einops import rearrange
from PIL import Image
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, AutoModelForSeq2SeqLM, MarianTokenizer

sys.path.append(os.path.dirname(__file__) + "/../../submodules/ru_dalle")
sys.path.append(os.path.dirname(__file__) + "/../../submodules/VQGAN")
from rudalle import get_rudalle_model, get_tokenizer, get_vae
from rudalle.dalle import MODELS
from rudalle.dalle.fp16 import FP16Module
from rudalle.dalle.model import DalleModel
from rudalle.dalle.utils import exists, is_empty

from . import SURREALIST_XL_DICT
from .generate import get_col_mask, get_conv_mask, get_image_pos_embeddings, get_row_mask

MODELS.update({"Surrealist_XL": SURREALIST_XL_DICT})


def infiniter(dataloader):
    while True:
        for sample in dataloader:
            yield sample


class RuDalleDataset(Dataset):
    def __init__(self, images, captions, height, width, stretch, random_crop, vae, tokenizer, text_seq_length, device):
        self.text_seq_length = text_seq_length
        self.tokenizer = tokenizer
        self.device = device
        self.token_shape = (height // 8, width // 8)

        self.image_transform = T.Compose(
            ([T.RandomCrop(random_crop, pad_if_needed=True, padding_mode="reflect")] if random_crop else [])
            + [
                T.Resize((height, width), antialias=True)
                if stretch
                else T.Compose([T.Resize(max(height, width), antialias=True), T.CenterCrop((height, width))])
            ]
            + [T.ToTensor()]
        )

        mname = "Helsinki-NLP/opus-mt-en-ru"
        translate_tokenizer = MarianTokenizer.from_pretrained(mname)
        translator = AutoModelForSeq2SeqLM.from_pretrained(mname)

        if len(captions) == 0:
            text_tokens = [self.tokenizer.encode_text("", text_seq_length=self.text_seq_length).squeeze(0)] * len(
                images
            )
        else:
            text_tokens = []
            for text in tqdm(captions, desc="preprocessing captions..."):
                translate_tokens = translator.generate(translate_tokenizer.encode(text, return_tensors="pt"))[0]
                translated = translate_tokenizer.decode(translate_tokens, skip_special_tokens=True)
                tokens = self.tokenizer.encode_text(translated, text_seq_length=self.text_seq_length).squeeze(0)
                text_tokens.append(tokens)

        image_tokens = []
        for image in tqdm(images, desc="preprocessing images..."):
            image = self.image_transform(Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image)
            tokens = vae.get_codebook_indices(image.unsqueeze(0).to(self.device)).cpu().squeeze(0)
            image_tokens.append(tokens)

        self.samples = list(zip(image_tokens, text_tokens))

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
    lr=1e-5,
    steps=500,
    batch_size=1,
    train_text=False,
    gradient_checkpointing=False,
    save_dir="modelzoo/",
    adam8bit=False,
):
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    text_seq_length, device = model.get_param("text_seq_length"), model.get_param("device")
    h, w = dataset.token_shape
    model.module.total_seq_length = total_seq_length = text_seq_length + w * h
    if w != model.module.image_col_embeddings.weight.shape[0] or h != model.module.image_row_embeddings.weight.shape[0]:
        model.module.image_tokens_per_dim = -1  # unused if everything is set up correctly
        model.module.image_seq_length = w * h
        model.module.transformer.row_mask = get_row_mask(text_seq_length, w, h, is_bool_mask=True).to(device)
        model.module.transformer.col_mask = get_col_mask(text_seq_length, w, h, is_bool_mask=True).to(device)
        model.module.transformer.conv_mask = get_conv_mask(text_seq_length, w, h, is_bool_mask=True).to(device)
        model.module.image_row_embeddings.weight = torch.nn.Parameter(
            interpolate(model.module.image_row_embeddings.weight.T.unsqueeze(0), h).squeeze(0).T
        )
        model.module.image_col_embeddings.weight = torch.nn.Parameter(
            interpolate(model.module.image_col_embeddings.weight.T.unsqueeze(0), w).squeeze(0).T
        )
        model.module.get_image_pos_embeddings = partial(get_image_pos_embeddings, dalle=model.module, width=w)

    model.train()
    model = freeze(model=model, freeze_emb=False, freeze_ln=False, freeze_attn=True, freeze_ff=True, freeze_other=False)

    optimizer = (Adam8bit if adam8bit else AdamW)(model.parameters(), lr=lr)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, final_div_factor=500, steps_per_epoch=int(np.ceil(steps / 30)), epochs=30
    )

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
                    t._get_layer_mask(i)[: attention_mask.size(2), : attention_mask.size(3)],
                ),
                use_cache=False,
            )
        )
    if gradient_checkpointing:
        embeddings = torch.utils.checkpoint.checkpoint_sequential(layers, 6, embeddings)
        transformer_output = embeddings
    else:
        hidden_states = embeddings
        for i in range(len(t.layers)):
            mask = torch.mul(attention_mask, t._get_layer_mask(i)[: attention_mask.size(2), : attention_mask.size(3)])
            hidden_states, cache = t.layers[i](hidden_states, mask, use_cache=use_cache)
        transformer_output = hidden_states
    transformer_output = self.transformer.final_layernorm(transformer_output)

    logits = self.to_logits(transformer_output)
    if return_loss is False:
        return logits, cache

    labels = torch.cat((text[:, 1:], image_input_ids), dim=1).contiguous().long()
    logits = rearrange(logits, "b n c -> b c n")

    text_logits = logits[:, : self.vocab_size, : self.text_seq_length].contiguous().float()
    image_logits = logits[:, self.vocab_size :, self.text_seq_length :].contiguous().float()

    loss_text = F.cross_entropy(text_logits, labels[:, : self.text_seq_length])
    loss_img = F.cross_entropy(image_logits, labels[:, self.text_seq_length :])

    loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
    return loss, {"text": loss_text.data.detach().float(), "image": loss_img}


def finetune(
    images: List[Union[str, Path, Image.Image]],
    captions: List[str] = [],
    model_name="rudalle",
    steps=500,
    lr=1e-5,
    batch_size=1,
    height=256,
    width=256,
    stretch=False,
    random_crop=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    low_memory=False,
    adam8bit=False,
    checkpoint: Optional[Union[str, Path]] = None,
    save_dir="modelzoo/",
) -> Tuple[Union[FP16Module, DalleModel], Optional[Tuple[int, int]]]:
    f"""Finetune a RuDALL-E model on a set of images (and possibly captions).

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
        random_crop (bool, optional): Randomly crop sections of this size during training. None disables random cropping.
        device (torch.device, optional): The device to train on, using 'cpu' will take a long time!. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        low_memory (bool, optional): Enable if you have less than 16 GB of (V)RAM to use gradient checkpointing (slower but more memory efficient). Defaults to False.
        checkpoint (Optional[Union[str, Path]], optional): Checkpoint to resume from. Either a path to a trained RuDALL-E checkpoint or one of {list(MODELS.keys())}.
        save_dir (str, optional): Directory to save finetuned checkpoints in. Defaults to "modelzoo/".

    Returns:
        Tuple[Union[FP16Module, DalleModel], Optional[Tuple[int, int]]]: The finetuned model and the size that sampled images should be stretched to if `stretch` is enabled.
    """

    assert len(captions) == 0 or len(captions) == len(
        images
    ), "When specifying captions, the number of images must match exactly."

    low_mem = low_memory

    stretched_size = None
    if stretch:
        original_size = (Image.open(images[0]) if isinstance(images[0], (str, Path)) else images[0]).size
        w, h = original_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = min(w, h, 256)
        new_short, new_long = requested_new_short, int(requested_new_short * long / short)
        stretched_size = (new_short, new_long) if w <= h else (new_long, new_short)

    if checkpoint is None:
        checkpoint = "Malevich"
    if checkpoint in list(MODELS.keys()):
        model = get_rudalle_model(checkpoint, pretrained=True, fp16=True, device=device, cache_dir="modelzoo/")
    else:
        model = get_rudalle_model("Malevich", pretrained=False, fp16=True, device=device, cache_dir="modelzoo/")
        ckpt = torch.load(checkpoint)
        model.module.image_row_embeddings.weight = torch.nn.Parameter(
            torch.zeros_like(ckpt["image_row_embeddings.weight"])
        )
        model.module.image_col_embeddings.weight = torch.nn.Parameter(
            torch.zeros_like(ckpt["image_col_embeddings.weight"])
        )
        model.module.transformer.row_mask = torch.zeros_like(ckpt["transformer.row_mask"])
        model.module.transformer.col_mask = torch.zeros_like(ckpt["transformer.col_mask"])
        model.module.transformer.conv_mask = torch.zeros_like(ckpt["transformer.conv_mask"])
        model.module.image_seq_length = (
            model.module.image_col_embeddings.weight.shape[0] * model.module.image_row_embeddings.weight.shape[0]
        )
        model.module.total_seq_length = model.module.text_seq_length + model.module.image_seq_length
        model.load_state_dict(ckpt)
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
            random_crop=random_crop,
            vae=get_vae(cache_dir="modelzoo/").to(device),
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
        adam8bit=adam8bit,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return model, stretched_size


def argument_parser():
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with images to train on")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of example images from input_dir to train on (None => All images will be used)")
    parser.add_argument("--input_imgs", type=list, default=[], nargs="*", help="List of images to train on")
    parser.add_argument("--captions", type=list, default=[], nargs="*", help="List of a caption for each image")
    parser.add_argument("--input_text", type=str, default="", help="Input text to sample images for after training. Will only have effect if you train with low number of steps / low learning rate or train with captions.")
    parser.add_argument("--num_outputs", type=int, default=8, help="Number of images to generate after finetuning")
    parser.add_argument("--steps", type=int, default=500, help="Number of batches to finetune for. More steps will converge to outputs that are closer to the inputs (which also means less variation).")
    parser.add_argument("--lr", type=float, default=1e-5, help="Starting learning rate (decays by a factor of 500 over training). A high learning rate will converge faster (which also means less variation). 1e-4 to 1e-5 is a good starting range (with 1e-5 resulting in more varied outputs).")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of images for each training step. Higher batch size requires more memory, but will be faster per sample overall.")
    parser.add_argument("--inference_batch_size", type=int, default=1, help="Number of images to sample at once. Higher batch size requires more memory, but will be faster per sample overall. Inference batches can be bigger as we don't need to store gradients for training.")
    parser.add_argument("--size", type=str, default='256,256', help="width,height of images to generate")
    parser.add_argument("--stretch", action="store_true", help="Squash images down to fixed size for training and stretch back to original size after sampling. This can significantly improve training/sampling time while still yielding good results. All training images should have the same aspect ratio.")
    parser.add_argument("--random_crop", type=int, default=None, help="Randomly crop sections of this size during training. None disables random cropping.")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2, 4, 8], help="Use RealESRGAN to upscale outputs.")
    parser.add_argument("--top_p", type=float, default=0.99, help="Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to train on, using 'cpu' will take a long time!")
    parser.add_argument("--low_memory", action="store_true", help="Enable if you have less than 16 GB of (V)RAM to use gradient checkpointing (slower but more memory efficient)")
    parser.add_argument("--adam8bit", action="store_true", help="Enable for even more memory-efficient training.")
    parser.add_argument("--checkpoint", type=str, default=None, help=f"Checkpoint to resume from. Either a path to a trained RuDALL-E checkpoint or one of {list(MODELS.keys())}.")
    parser.add_argument("--save_dir", type=str, default="modelzoo/", help="Directory to save finetuned checkpoints in.")
    parser.add_argument("--model_name", type=str, default=None, help="Name for finetuned checkpoints. Will default to the name of input_dir or the first input_img.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save output images in.")
    # fmt: on
    return parser


def main(args):
    assert len(args.captions) == 0 or len(args.input_imgs) == len(
        args.captions
    ), "When specifying captions, the number of input_imgs must match exactly."
    assert (
        len(args.input_imgs) > 0 or args.input_dir is not None
    ), "You must specify either a list of input_imgs or an input_dir to train with."

    images = args.input_imgs
    if len(images) == 0:
        images = glob(args.input_dir + "/*", recursive=True)
        if args.num_examples is not None:
            images = random.choices(images, k=args.num_examples)

    width, height = [int(v) for v in args.size.split(",")]
    stem = Path(args.input_dir).stem if args.input_dir is not None else Path(args.input_imgs[0]).stem
    model_name = f"{stem}_{width}x{height}_lr={args.lr}_steps={args.steps}_rudalle_finetuned"

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
        random_crop=args.random_crop,
        device=torch.device(args.device),
        low_memory=args.low_memory,
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
    )

    from maua.autoregressive.ru_dalle.generate import generate

    outputs = generate(
        model.eval(),
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
        oversample=False,
    )
    for id, im in enumerate(outputs):
        im.save(f"{args.output_dir}/{model_name}_{id}.png")


if __name__ == "__main__":
    main(argument_parser().parse_args())
