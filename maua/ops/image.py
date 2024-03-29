import math
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from medpy.filter.noise import immerkaer as immerkaer_np
from PIL import Image
from resize_right import resize
from scipy.special import comb
from torchvision.transforms.functional import adjust_sharpness


def destitch(img, tile_size, overtile=1):
    _, _, H, W = img.shape
    n_rows = round(np.floor(H / tile_size) + overtile)
    n_cols = round(np.floor(W / tile_size) + overtile)
    tiled = []
    for y in torch.linspace(0, H - tile_size, n_rows).round().long():
        for x in torch.linspace(0, W - tile_size, n_cols).round().long():
            tiled.append(img[..., y : y + tile_size, x : x + tile_size])
    return torch.cat(tiled, dim=0)


def smoothstep(x, N=2):
    result = torch.zeros_like(x)
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    return result


def blend_weight1d(total_size, fade_in, fade_out):
    return torch.cat(
        (
            smoothstep(torch.linspace(0, 1, fade_in)),
            torch.ones(total_size - fade_in - fade_out),
            smoothstep(torch.linspace(1, 0, fade_out)),
        )
    )


def restitch(tiled, H, W, overtile=1):
    _, C, _, tile_size = tiled.shape
    n_rows = round(np.floor(H / tile_size) + overtile)
    n_cols = round(np.floor(W / tile_size) + overtile)
    out = torch.zeros((1, C, H, W), device=tiled.device)
    rescale = torch.zeros_like(out)  # required to ensure rounding errors don't cause blending artifacts
    i = 0
    ys = torch.linspace(0, H - tile_size, n_rows).round().long()
    xs = torch.linspace(0, W - tile_size, n_cols).round().long()
    fade = tile_size - ys[1]
    for y in ys:
        wy = blend_weight1d(tile_size, fade_in=0 if y == 0 else fade, fade_out=0 if y == ys[-1] else fade).to(tiled)
        for x in xs:
            wx = blend_weight1d(tile_size, fade_in=0 if x == 0 else fade, fade_out=0 if x == xs[-1] else fade).to(tiled)
            weight = wy.reshape(1, 1, -1, 1) * wx.reshape(1, 1, 1, -1)
            out[..., y : y + tile_size, x : x + tile_size] += tiled[i] * weight
            rescale[..., y : y + tile_size, x : x + tile_size] += weight
            i += 1
    return out / rescale


def immerkaer(image_batch):  # TODO translate to pytorch
    sigmas = torch.tensor([immerkaer_np(image.permute(1, 2, 0).cpu().numpy()) for image in image_batch])
    return sigmas.to(image_batch)


def sharpen(img, strength):
    return adjust_sharpness(img.add(1).div(2), strength).mul(2).sub(1)


def local_std(im, ks=9):
    r = ks // 2
    im = F.pad(im, (r, r, r, r), mode="reflect")
    kernel = torch.ones((1, 1, ks, ks), device=im.device, dtype=im.dtype).tile(1, im.shape[1], 1, 1)
    s = F.conv2d(im, kernel)
    s2 = F.conv2d(im**2, kernel)
    ns = F.conv2d(torch.ones_like(im), kernel)
    return torch.sqrt((s2 - s**2 / ns) / ns)


def original_colors(content, generated):
    # Combine the Y channel of the generated image and the UV/CbCr channels of the
    # content image to perform color-independent style transfer.
    content_channels = list(content.resize(generated.size).convert("YCbCr").split())
    generated_channels = list(generated.convert("YCbCr").split())
    content_channels[0] = generated_channels[0]
    return Image.merge("YCbCr", content_channels).convert("RGB")


def wrapping_slice(tensor, start, length, return_indices=False):
    if start + length <= tensor.shape[0]:
        indices = torch.arange(start, start + length)
    else:
        indices = torch.cat((torch.arange(start, tensor.shape[0]), torch.arange(0, (start + length) % tensor.shape[0])))
    if tensor.shape[0] == 1:
        indices = torch.zeros(1, dtype=torch.int64)
    if return_indices:
        return indices
    return tensor[indices]


def get_histogram(tensor):
    mu_h = tensor.mean(list(range(len(tensor.shape) - 1)))
    h = tensor - mu_h
    h = h.permute(0, 3, 1, 2).reshape(tensor.size(3), -1)
    Ch = torch.mm(h, h.T) / h.shape[1] + torch.finfo(tensor.dtype).eps * torch.eye(h.shape[0], device=tensor.device)
    return mu_h, h, Ch


def match_histogram(target_tensor, source_tensor, mode="avg"):
    if mode == "False":
        return target_tensor

    backup = target_tensor.clone()
    try:
        if mode == "avg":
            elementwise = True
            random_frame = False
        else:
            elementwise = False
            random_frame = True

        if not isinstance(source_tensor, list):
            source_tensor = [source_tensor]

        output_tensor = torch.zeros_like(target_tensor)
        for source in source_tensor:
            target = target_tensor.permute(0, 3, 2, 1)  # Function expects b,w,h,c
            source = source.permute(0, 3, 2, 1)  # Function expects b,w,h,c
            if elementwise:
                source = source.mean(0).unsqueeze(0)
            if random_frame:
                source = source[np.random.randint(0, source.shape[0])].unsqueeze(0)

            matched_tensor = torch.zeros_like(target)
            for idx in range(target.shape[0] if elementwise else 1):
                frame = target[idx].unsqueeze(0) if elementwise else target
                _, t, Ct = get_histogram(frame + 1e-3 * torch.randn(size=frame.shape, device=frame.device))
                mu_s, _, Cs = get_histogram(
                    source.to(frame) + 1e-3 * torch.randn(size=source.shape, device=frame.device)
                )

                # PCA
                eva_t, eve_t = torch.linalg.eigh(Ct, UPLO="U")
                Et = torch.sqrt(torch.diagflat(eva_t))
                Et[Et != Et] = 0  # Convert nan to 0
                Qt = torch.mm(torch.mm(eve_t, Et), eve_t.T)

                eva_s, eve_s = torch.linalg.eigh(Cs, UPLO="U")
                Es = torch.sqrt(torch.diagflat(eva_s))
                Es[Es != Es] = 0  # Convert nan to 0
                Qs = torch.mm(torch.mm(eve_s, Es), eve_s.T)

                ts = torch.mm(torch.mm(Qs, torch.inverse(Qt)), t)

                match = ts.reshape(*frame.permute(0, 3, 1, 2).shape).permute(0, 2, 3, 1)
                match += mu_s

                if elementwise:
                    matched_tensor[idx] = match
                else:
                    matched_tensor = match
            output_tensor += matched_tensor.permute(0, 3, 2, 1) / len(source_tensor)
    except RuntimeError:
        traceback.print_exc()
        print("Skipping histogram matching...")
        output_tensor = backup
    return output_tensor.clamp(
        min([s.min().item() for s in source_tensor]), max([s.max().item() for s in source_tensor])
    )


def luminance(tensor):
    return 0.2126 * tensor[:, :, 0] + 0.7152 * tensor[:, :, 1] + 0.0722 * tensor[:, :, 2]


def color_balance(img, percent):
    """From https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc"""
    out_channels = []
    cumstops = (img.shape[0] * img.shape[1] * percent / 200.0, img.shape[0] * img.shape[1] * (1 - percent / 200.0))
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate(
            (np.zeros(low_cut), np.around(np.linspace(0, 255, high_cut - low_cut + 1)), 255 * np.ones(255 - high_cut))
        )
        out_channels.append(cv2.LUT(channel, lut.astype("uint8")))
    return cv2.merge(out_channels)


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape

    if isinstance(size, (int, float)):
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size
        new_short, new_long = round(requested_new_short), round(requested_new_short * long / short)
        dw, dh = (new_short, new_long) if w <= h else (new_long, new_short)
    else:
        dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, (dh, dw), mode="bicubic", align_corners=align_corners)


def unsharp_mask(img, ks=(7, 7), sigma=1.0, amount=1, thresh=0.25):
    blurred = cv2.GaussianBlur(img, ks, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if thresh > 0:
        low_contrast_mask = np.absolute(img - blurred) < thresh
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened


def normalize_np(img):
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype("float") - min_val) / (max_val - min_val)
    return out


def positive(x):
    return (x > 0).astype(float)


def blurriness_lbp(im_gray, ks, thresh):
    I = normalize_np(im_gray)
    pt = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    right, left, above, below = pt[1:-1, 2:], pt[1:-1, :-2], pt[:-2, 1:-1], pt[2:, 1:-1]
    aboveRight, aboveLeft, belowRight, belowLeft = pt[:-2, 2:], pt[:-2, :-2], pt[2:, 2:], pt[2:, :-2]

    Q = math.sqrt(2) / 2  # interp offset
    interp1 = (1 - Q) * ((1 - Q) * I + Q * right) + Q * ((1 - Q) * above + Q * aboveRight)
    interp3 = (1 - Q) * ((1 - Q) * I + Q * left) + Q * ((1 - Q) * above + Q * aboveLeft)
    interp5 = (1 - Q) * ((1 - Q) * I + Q * left) + Q * ((1 - Q) * below + Q * belowLeft)
    interp7 = (1 - Q) * ((1 - Q) * I + Q * right) + Q * ((1 - Q) * below + Q * belowRight)

    s0 = positive(right - I - thresh)
    s1 = positive(interp1 - I - thresh)
    s2 = positive(above - I - thresh)
    s3 = positive(interp3 - I - thresh)
    s4 = positive(left - I - thresh)
    s5 = positive(interp5 - I - thresh)
    s6 = positive(below - I - thresh)
    s7 = positive(interp7 - I - thresh)

    U = (
        np.abs(s0 - s7)
        + np.abs(s1 - s0)
        + np.abs(s2 - s1)
        + np.abs(s3 - s2)
        + np.abs(s4 - s3)
        + np.abs(s5 - s4)
        + np.abs(s6 - s5)
        + np.abs(s7 - s6)
    )
    lbpmap = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
    lbpmap[U > 2] = 9

    window_r = (ks - 1) // 2
    h, w = im_gray.shape[:2]
    sharp_map = np.zeros((h, w), dtype=float)
    lbpmap_pad = cv2.copyMakeBorder(lbpmap, window_r, window_r, window_r, window_r, cv2.BORDER_REPLICATE)

    lbpmap_sum = (
        (lbpmap_pad == 6).astype(float)
        + (lbpmap_pad == 7).astype(float)
        + (lbpmap_pad == 8).astype(float)
        + (lbpmap_pad == 9).astype(float)
    )
    integral = cv2.integral(lbpmap_sum)
    integral = integral.astype(float)

    sharp_map = (
        integral[ks - 1 : -1, ks - 1 : -1]
        - integral[0:h, ks - 1 : -1]
        - integral[ks - 1 : -1, 0:w]
        + integral[0:h, 0:w]
    ) / math.pow(ks, 2)

    return sharp_map


def windowed_index(height: int, width: int, seg_size: int):
    ys = torch.arange(height, dtype=torch.long, device=f"cuda")
    xs = torch.arange(width, dtype=torch.long, device=f"cuda")
    ys, xs = torch.meshgrid(ys, xs)
    idxs = torch.stack([ys.flatten(), xs.flatten()])

    winrange = torch.arange(seg_size, dtype=torch.long, device=f"cuda")
    ywin, xwin = torch.meshgrid(winrange, winrange)
    window = torch.stack((ywin, xwin))

    idxs = idxs[:, :, None, None] + window[:, None, :, :]

    return idxs[0].clamp(0, height - 1), idxs[1].clamp(0, width - 1)


def blurriness_svd(img, kr=10, sv_num=3):
    h, w = img.shape

    img = torch.nn.functional.pad(torch.from_numpy(img).float().cuda()[None, None], [kr] * 4, mode="reflect").squeeze()

    ys, xs = windowed_index(h, w, kr * 2)
    blocks = img[ys, xs]

    _, s, _ = torch.svd(blocks)

    top_svs = torch.sum(s[:, 0:sv_num], axis=1)
    total_svs = torch.sum(s, axis=1)
    sv_degrees = top_svs / total_svs
    max_sv = sv_degrees.min()
    min_sv = sv_degrees.max()

    blur_map = (sv_degrees - min_sv) / (max_sv - min_sv)

    return blur_map.float().reshape(h, w).cpu().numpy()


def scaled_height_width(h, w, size):
    short, long = (w, h) if w <= h else (h, w)
    requested_new_short = size
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    w, h = (new_short, new_long) if w <= h else (new_long, new_short)
    return math.ceil(h / 2.0) * 2, math.ceil(w / 2.0) * 2
