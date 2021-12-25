"""
Based on https://github.com/znah/gitart_kunstformen_ca
"""

from NCA_train import *
from glob import glob

model_file = f"{out_dir}{name(style_file)}_7500.pt"
ca = torch.load(model_file)
num_frames = 600

# NCA evolution video
with VideoWriter(f"{out_dir}/{name(style_file)}_{name(model_file).split('_')[-1]}.mp4") as vid, torch.no_grad():
    x = ca.seed(1, 256)
    for k in tqdm(range(num_frames)):
        step_n = min(2 ** (k // 30), 32)
        for i in range(step_n):
            x[:] = ca(x)
        img = to_rgb(x[0]).permute(1, 2, 0).cpu()
        vid.add(zoom(img, 2))

# Running all training checkpoints on the columns of the same grid
# Early model snapshots aren't stable in the long term.
with VideoWriter(f"{out_dir}/{name(style_file)}_checkgrid.mp4") as vid, torch.no_grad():
    models = [torch.load(fn) for fn in sorted(glob(f"{out_dir}/{name(style_file)}*.pt"))][2:-2]
    chn = models[0].chn
    w = 128
    x = torch.rand(1, chn, 512, w * len(models) + 2) * 0.1
    for k in tqdm(range(num_frames)):
        for i in range(8):
            for ci, f in enumerate(models):
                sub = x[:, :, :, ci * w : ci * w + w + 2]
                sub[:, :, :, 1:-1] = f(sub)[:, :, :, 1:-1]
        img = to_rgb(x[0]).permute(1, 2, 0).cpu()
        vid.add(zoom(img, 2))

s = "W\u039BV"
font = PIL.ImageFont.truetype("DejaVuSans.ttf", 256)
w, h = font.getsize(s)
print(w, h)
pad = 64
im = PIL.Image.new("L", (w + pad * 2, h + pad * 2))
draw = PIL.ImageDraw.Draw(im)
draw.text((pad, pad), s, fill=255, font=font)
im = im.filter(PIL.ImageFilter.GaussianBlur(5))
p = np.float32(im)
p = p / p.max() * 0.6 + 0.05
p = torch.tensor(p)

with VideoWriter(f"{out_dir}/{name(style_file)}-{name(model_file).split('_')[-1]}-wav.mp4") as vid, torch.no_grad():
    h, w = p.shape
    x = torch.zeros([1, ca.chn, h, w])
    for k in tqdm(range(num_frames)):
        step_n = min(int(2 ** (k / 30)), 32)
        for i in range(step_n):
            x[:] = ca(x, p)
        img = to_rgb(x[0]).permute(1, 2, 0).cpu()
        img *= min(1.0 - (k - 400) / 100, 1.0)  # fade out
        vid.add(zoom(img, 2))
