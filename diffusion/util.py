import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)

        if sideY < sideX:
            size = sideY
            tops = torch.zeros(self.cutn // 4, dtype=int)
            lefts = torch.linspace(0, sideX - size, self.cutn // 4, dtype=int)
        else:
            size = sideX
            tops = torch.linspace(0, sideY - size, self.cutn // 4, dtype=int)
            lefts = torch.zeros(self.cutn // 4, dtype=int)

        cutouts = []
        for offsety, offsetx in zip(tops, lefts):
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        for _ in range(self.cutn - len(cutouts)):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            loc = torch.randint(0, (sideX - size + 1) * (sideY - size + 1), ())
            offsety, offsetx = torch.div(loc, (sideX - size + 1), rounding_mode="floor"), loc % (sideX - size + 1)
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def parse_prompt(prompt):
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])
