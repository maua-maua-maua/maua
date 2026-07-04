# %%
import base64
import colorsys
import io
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rich_pixels
from openai import OpenAI
from PIL import Image
from rich.console import Console

# "/home/hans/ModelZoo/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-Q8_0.gguf"
# "/home/hans/ModelZoo/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/mmproj-BF16.gguf"


def image_to_base64_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64_data}"


def get_palette(image: Image.Image) -> list[tuple[int, int, int]]:
    # Ensure we always have RGBA tuples
    rgba = image.convert("RGBA")
    # Unique colors as (R, G, B, A)
    unique = {(r, g, b, a) for r, g, b, a in rgba.get_flattened_data()}
    # Transparent first (any fully transparent pixel, regardless of RGB)
    transparent = [c for c in unique if c[3] == 0]
    opaque = [c[:3] for c in unique if c[3] != 0]

    # Sort opaque colors by hue, then saturation, then value, then alpha
    def hls_key(c):
        r, g, b = c
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        return (h, l, s)

    opaque_sorted = sorted(opaque, key=hls_key)
    # Keep exactly one transparent entry at the front if any transparency exists
    if transparent:
        return [(255, 0, 255)] + opaque_sorted
    return opaque_sorted


def palette_to_image(palette: list[tuple[int, int, int]]) -> Image.Image:
    palette_img = Image.new("RGBA", (max(1, len(palette)), 1))
    palette_img.putdata(palette)
    palette_img = palette_img.resize((max(1, len(palette)) * 2, 2), Image.Resampling.NEAREST)
    return palette_img


USER_PROMPTS = [
    "Describe the image in detail please. Filename: {filename}",
    "Give a short description of the image. Filename: {filename}",
    "Describe the image maximally four words. Filename: {filename}",
    "Describe the image in one word. Filename: {filename}",
    "Describe the image. Filename: {filename}",
]


def get_caption(image: Image.Image, prompt: str) -> str:
    data_uri = image_to_base64_data_uri(image.resize((image.width * 4, image.height * 4)))

    response = client.chat.completions.create(
        model="pixel-caption",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that describes pixel art images and GBA game sprites. There is no need to describe the style, just give a simple description of the objects in the image. Transparent backgrounds will be filled in with black, do not mention this in your description. Do not mention 'the image ...', 'this image ...', etc. in your description. Avoid any markdown or formatting in your description. Phrase the description as if the user is prompting a model to generate an image based on the description.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        stream=False,
    )
    return str(response.choices[0].message.content)


if __name__ == "__main__":
    base_dir = Path("/home/hans/Pictures/pixlm")
    im_paths = list(base_dir.glob("objects/*.png"))

    # %%
    client = OpenAI(
        base_url="http://localhost:6006/v1",
        api_key=os.getenv("UNSLOTH_API_KEY"),
    )

    console = Console()

    for path in im_paths:
        image = Image.open(path)
        palette = get_palette(image)
        palette_img = palette_to_image(palette)
        with ThreadPoolExecutor(max_workers=5) as executor:
            captions = list(
                executor.map(
                    get_caption,
                    [image] * len(USER_PROMPTS),
                    [prompt.format(filename=path.name) for prompt in USER_PROMPTS],
                )
            )

        with open(path.with_suffix(".json"), "w") as f:
            json.dump({"captions": captions}, f)

        console.print()
        console.print(rich_pixels.Pixels.from_image(image))
        console.print(path)
        console.print(f"Palette: ({len(palette)} colors)")
        console.print(rich_pixels.Pixels.from_image(palette_img))
        console.print("Captions:")
        for caption in captions:
            console.print(f"- {caption}")
        console.print()

    # %%

    ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    IMAGE_FORMAT = """
<palette>
{palette}
</palette>
<image>
{image}
</image>
""".strip()

    def image_to_string(image: Image.Image) -> str:
        palette = get_palette(image)
        palette[0] = ("-", "-", "-")
        palette_str = "\n".join([f"{ALPHABET[c]}:{color[0]},{color[1]},{color[2]}" for c, color in enumerate(palette)])

        idxs = np.array([
            ALPHABET[0 if a == 0 else palette.index((r, g, b))] for (r, g, b, a) in image.get_flattened_data()
        ])
        idxs = idxs.reshape(image.height, image.width)
        idxs = idxs.astype(str).tolist()
        idx_str = "\n".join(["".join(row) for row in idxs])

        return IMAGE_FORMAT.format(palette=palette_str, image=idx_str)

    def string_to_image(string: str) -> Image.Image:
        palette_str = string.split("<palette>")[1].split("</palette>")[0].strip()
        palette = np.array([
            (*tuple(map(int, line.replace("-", "0").strip().split(":")[1].split(","))), 255)
            for line in palette_str.split("\n")
        ])
        palette[0] = (0, 0, 0, 0)

        image_str = string.split("<image>")[1].split("</image>")[0].strip()
        idxs = np.array([[ALPHABET.index(c) for c in row] for row in image_str.split("\n")])

        pixels = palette[idxs]
        image = Image.fromarray(pixels.astype(np.uint8))

        return image

    # %%
    dataset = []

    for path in im_paths:
        captions_path = path.with_suffix(".json")
        with open(captions_path, "r") as f:
            captions = json.load(f)["captions"]

        image = Image.open(path)

        for caption in captions:
            dataset.append({
                "messages": [
                    {"role": "user", "content": caption + "\n" + f"Size: {image.width}x{image.height}"},
                    {"role": "assistant", "content": image_to_string(image)},
                ]
            })

    # %%
    train_split = int(0.9 * len(im_paths) * len(USER_PROMPTS))
    train_dataset = dataset[:train_split]
    val_dataset = dataset[train_split:]
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)

    with open(base_dir / "train.jsonl", "w") as f:
        for item in train_dataset:
            f.write(json.dumps(item) + "\n")

    with open(base_dir / "val.jsonl", "w") as f:
        for item in val_dataset:
            f.write(json.dumps(item) + "\n")


# %%
# if __name__ == "__main__":
#     import torch
#     from unsloth import FastLanguageModel

#     # Load model with QLoRA (4-bit)
#     model_name = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
#     model_stem = Path(model_name).stem
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=model_name,
#         max_seq_length=4096,
#         dtype=None,  # Auto-detect
#         load_in_4bit=True,  # QLoRA
#     )

#     # Add LoRA adapters
#     model = FastLanguageModel.get_peft_model(
#         model,
#         r=8,  # Rank — higher = more capacity, more VRAM
#         target_modules=[
#             "q_proj",
#             "k_proj",
#             "v_proj",
#             "o_proj",
#             "gate_proj",
#             "up_proj",
#             "down_proj",
#         ],
#         lora_alpha=16,
#         lora_dropout=0,  # Unsloth optimized — keep at 0
#         bias="none",
#         use_gradient_checkpointing="unsloth",
#     )

#     # Load training data
#     from datasets import load_dataset

#     dataset = load_dataset("json", data_files=str(base_dir / "train.jsonl"), split="train")

#     #%%
#     # Training arguments
#     from datetime import datetime

#     from transformers import TrainingArguments
#     from trl import SFTTrainer

#     now = int(datetime.now().strftime("%Y%m%d%H%M"))
#     train_dir = base_dir / f"runs/{model_stem}-{now}"

#     trainer = SFTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         train_dataset=dataset,
#         max_seq_length=4096,
#         args=TrainingArguments(
#             per_device_train_batch_size=2,
#             gradient_accumulation_steps=4,
#             warmup_steps=10,
#             num_train_epochs=3,
#             learning_rate=2e-4,
#             fp16=not torch.cuda.is_bf16_supported(),
#             bf16=torch.cuda.is_bf16_supported(),
#             logging_steps=10,
#             output_dir=str(train_dir),
#             optim="adamw_8bit",
#             seed=(now * 42) % 2**32,
#         ),
#     )

#     # Train
#     trainer.train()

#     # Save LoRA adapter
#     model.save_pretrained(str(train_dir / f"{model_stem}-pixeLM"))
#     tokenizer.save_pretrained(str(train_dir / f"{model_stem}-pixeLM"))
#     for method in ["q4_k_m", "q5_k_m", "q8_0"]:
#         model.save_pretrained_gguf(
#             str(train_dir / f"{model_stem}-pixeLM-gguf-{method}"),
#             tokenizer,
#             quantization_method=method,
#         )
