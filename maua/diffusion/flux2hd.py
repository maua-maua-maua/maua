# %%
from pathlib import Path
from uuid import uuid4

import PIL.Image
import torch
import unfake
from diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint import Flux2KleinInpaintPipeline
from diffusers.utils.loading_utils import load_image
from IPython.display import display
from tqdm.auto import tqdm

repo_id = "black-forest-labs/FLUX.2-klein-9B"
torch_dtype = torch.bfloat16

pipe = Flux2KleinInpaintPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype)
pipe.transformer.set_attention_backend("sage")
pipe.enable_model_cpu_offload()

# %%
factor = 4

tileset = load_image("/home/hans/Pictures/moai/tropical.png")
tileset = tileset.resize((tileset.width * factor, tileset.height * factor), resample=PIL.Image.Resampling.NEAREST)

source = load_image("/home/hans/Pictures/moai/jungle.png")
source = source.resize((source.width * factor, source.height * factor), resample=PIL.Image.Resampling.NEAREST)

mask = PIL.Image.new("RGB", (source.width, source.height), "white")

name = "jungle"

prompt = "Make the image match the flat GBA tileset style of the reference image more closely. It should respect the 16x16 pixel grid and use a small palette of colors with less shading."

for _ in range(5):
    im = pipe.__call__(
        prompt=prompt,
        image=source,
        image_reference=tileset,
        mask_image=mask,
        strength=0.8,
        width=source.width,
        height=source.height,
    ).images[0]

    uid = str(uuid4())[:6]
    file = f"/home/hans/Pictures/moai/{name}_{uid}.png"
    display(im)
    im.save(file)

# %%
# image_paths = list(Path("/home/hans/neurout/autoregressive/whiopt").glob("*.png"))

# output_dir = Path("/home/hans/neurout/autoregressive/whiopt_flux2hd")
# output_dir.mkdir(parents=True, exist_ok=True)

# with tqdm(total=len(image_paths) * 5) as pbar:
#     for n in range(5):
#         for image_path in image_paths:
#             pipe(
#                 prompt="",
#                 image=load_image(str(image_path)),
#                 num_inference_steps=50,
#                 width=1920,
#                 height=1080,
#             ).images[0].save(output_dir / f"{image_path.stem}_flux9hd_{n}.jpg")
#             pbar.update(1)

# %%
