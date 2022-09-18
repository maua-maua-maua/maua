import torch


class Noise(torch.nn.Module):
    def __init__(self, length, size):
        super().__init__()
        self.length = length
        self.size = size


class Blend(Noise):
    def __init__(self, rng, length, size, modulator):
        super().__init__(length, size)
        self.register_buffer(
            "noise", torch.randn((2, modulator.shape[1], size[0], size[1]), generator=rng, device=rng.device)
        )
        self.register_buffer("modulator", modulator)

    def forward(self, i, b):
        mod = self.modulator[i : i + b]
        mod = mod.reshape(len(mod), -1)
        left = torch.einsum("MHW,BM->BHW", self.noise[0], mod)
        right = torch.einsum("MHW,BM->BHW", self.noise[1], 1 - mod)
        return left + right


class Multiply(Noise):
    def __init__(self, rng, length, size, modulator):
        super().__init__(length, size)
        self.register_buffer(
            "noise", torch.randn((modulator.shape[1], size[0], size[1]), generator=rng, device=rng.device)
        )
        self.register_buffer("modulator", modulator)

    def forward(self, i, b):
        mod = self.modulator[i : i + b]
        mod = mod.reshape(len(mod), -1)
        left = torch.einsum("MHW,BM->BHW", self.noise, mod)
        return left


class Loop(Noise):
    def __init__(self, rng, length, size, n_loops=1, sigma=5):
        super().__init__(length, size)
        self.sigma = sigma
        self.register_buffer("noise", torch.randn((3, size[0], size[1]), generator=rng, device=rng.device))
        self.register_buffer("idx", torch.linspace(0, n_loops * 2 * torch.pi, length))

    def forward(self, i, b):
        freqs = torch.cos(self.idx[i : i + b, None, None] + self.noise[[0]]).div(self.sigma / 50)
        out = torch.sin(freqs + self.noise[[1]]) * self.noise[[2]]
        out = out / (out.square().mean(dim=(1, 2), keepdim=True).sqrt() + torch.finfo(out.dtype).eps)
        return out


class Average(Noise):
    def __init__(self, left, right):
        super().__init__(left.length, left.size)
        self.left = left
        self.right = right

    def forward(self, i, b):
        return (self.left(i, b) + self.right(i, b)) / 2


class Modulate(Noise):
    def __init__(self, left, right, modulator):
        super().__init__(left.length, left.size)
        self.left = left
        self.right = right
        self.register_buffer("modulator", modulator.mean(1))

    def forward(self, i, b):
        mod = self.modulator[i : i + b, None, None]
        return self.left(i, b) * mod + self.right(i, b) * (1 - mod)


class ScaleBias(Noise):
    def __init__(self, base, scale, bias):
        super().__init__(base.length, base.size)
        self.base = base
        self.scale = scale
        self.bias = bias

    def forward(self, i, b):
        return self.scale * self.base(i, b) + self.bias


def noise_patch(
    rng,
    noise,
    features,
    tempo,
    fps,
    patch_type,
    loop_bars,
    seq_feat,
    seq_feat_weight,
    mod_feat,
    mod_feat_weight,
    merge_type,
    merge_depth,
    noise_mean,
    noise_std,
):
    if merge_depth == "low":
        lays = range(0, 6)
    elif merge_depth == "mid":
        lays = range(6, 12)
    elif merge_depth == "high":
        lays = range(12, 17)
    elif merge_depth == "lowmid":
        lays = range(0, 12)
    elif merge_depth == "midhigh":
        lays = range(6, 17)
    elif merge_depth == "all":
        lays = range(0, 17)

    feature = seq_feat_weight * features[seq_feat]

    for n in lays:

        if patch_type == "blend":
            new_noise = Blend(rng=rng, length=len(feature), size=noise[n].size, modulator=feature)
        elif patch_type == "multiply":
            new_noise = Multiply(rng=rng, length=len(feature), size=noise[n].size, modulator=feature)
        elif patch_type == "loop":
            n_loops = len(feature) / fps / 60 / tempo / 4 / loop_bars
            new_noise = Loop(rng=rng, length=len(feature), size=noise[n].size, n_loops=n_loops)

        if merge_type == "average":
            noise[n] = Average(left=noise[n], right=new_noise)
        elif merge_type == "modulate":
            noise[n] = Modulate(left=noise[n], right=new_noise, modulator=mod_feat_weight * features[mod_feat])
        else:  # overwrite
            noise[n] = new_noise

        noise[n] = ScaleBias(noise[n], scale=noise_std, bias=noise_mean)

    return noise
