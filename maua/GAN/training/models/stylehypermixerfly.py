from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import interpolate
from torch_butterfly import Butterfly


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class GLU(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop=0.0, internal=True):
        super().__init__()
        self.internal = internal
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        assert hidden_dim % 2 == 0

        self.fc1 = Butterfly(in_dim, hidden_dim)
        self.act = nn.Sigmoid()
        self.fc2 = Butterfly(hidden_dim // 2, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        x = self.fc2(x)
        if self.internal:
            x = self.drop(x)
        return x


class HyperMixer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.mlp_1 = GLU(in_dim=in_dim, out_dim=hidden_dim, drop=drop)
        self.mlp_2 = GLU(in_dim=in_dim, out_dim=hidden_dim, drop=drop)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :return (torch.Tensor): Output tensor of the shape
        """
        # Compute weights
        w_1 = self.mlp_1(x)
        w_2 = self.mlp_2(x)
        # Map input with weights and activate
        x = self.drop(self.act(w_1.transpose(1, 2) @ x))
        x = self.drop(w_2 @ x)
        return x


class HyperMixerBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mlp_ratio: Tuple[float, float] = (0.5, 2.0),
        drop: float = 0.1,
        drop_path: float = 0.1,
    ) -> None:
        """
        Constructor method
        :param in_dim (int): Input channel dimension
        :param out_dim (int): Output channel dimension
        :param mlp_ratio (Tuple[int, int]): Ratio of hidden dim. of the hyper mixer layer and MLP. Default = (0.5, 2.0)
        :param drop (float): Dropout rate. Default = 0.1
        :param drop_path (float): Dropout path rate. Default = 0.1
        """
        super().__init__()
        tokens_dim, channels_dim = [int(x * in_dim) for x in mlp_ratio]
        self.norm1 = nn.LayerNorm(in_dim, eps=1e-06)
        self.mlp_tokens = HyperMixer(in_dim=in_dim, hidden_dim=tokens_dim, drop=drop)
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm2 = nn.LayerNorm(in_dim, eps=1e-06)
        self.mlp_channels = GLU(in_dim=in_dim, hidden_dim=channels_dim, drop=drop)
        self.mlp_reduce = GLU(in_dim=in_dim, out_dim=out_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :return (torch.Tensor): Output tensor of the shape [batch size, tokens, channels]
        """
        x = self.norm1(x)
        x = x + self.drop_path(self.mlp_tokens(x))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        x = self.mlp_reduce(x)
        return x


class StyleGLU(nn.Module):
    def __init__(self, w_dim, in_dim, hidden_dim=None, out_dim=None, drop=0.0, internal=True):
        super().__init__()
        self.internal = internal
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        assert hidden_dim % 2 == 0

        self.gate_fc = Butterfly(in_dim, hidden_dim)
        self.act = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
        self.style_fc = Butterfly(w_dim, hidden_dim // 2)
        self.register_buffer("weight", torch.randn(hidden_dim // 2, out_dim))
        self.register_buffer("bias", torch.randn(out_dim))

    def forward(self, x, w):
        x = self.gate_fc(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)

        x = self.drop(x)

        s = self.style_fc(w)
        weight = self.weight[None, :, :] * s[:, :, None]
        x = torch.matmul(x, weight) + self.bias

        if self.internal:
            x = self.drop(x)
        else:
            x = torch.tanh(x)

        return x


class StyleHyperMixer(nn.Module):
    def __init__(self, w_dim: int, in_dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        # Init modules
        self.mlp_1 = StyleGLU(w_dim=w_dim, in_dim=in_dim, out_dim=hidden_dim, drop=drop)
        self.mlp_2 = StyleGLU(w_dim=w_dim, in_dim=in_dim, out_dim=hidden_dim, drop=drop)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :return (torch.Tensor): Output tensor of the shape
        """
        # Compute weights
        w_1 = self.mlp_1(x, w)
        w_2 = self.mlp_2(x, w)
        # Map input with weights and activate
        x = self.drop(self.act(w_1.transpose(1, 2) @ x))
        x = self.drop(w_2 @ x)
        return x


class StyleHyperMixerBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w_dim: int,
        mlp_ratio: Tuple[float, float] = (0.5, 2.0),
        drop: float = 0.1,
        drop_path: float = 0.1,
    ) -> None:
        """
        Constructor method
        :param in_dim (int): Input channel dimension
        :param out_dim (int): Output channel dimension
        :param mlp_ratio (Tuple[int, int]): Ratio of hidden dim. of the hyper mixer layer and MLP. Default = (0.5, 2.0)
        :param drop (float): Dropout rate. Default = 0.1
        :param drop_path (float): Dropout path rate. Default = 0.1
        """
        super().__init__()
        tokens_dim, channels_dim = [int(x * in_dim) for x in mlp_ratio]
        self.norm1 = nn.LayerNorm(in_dim, eps=1e-06)
        self.mlp_tokens = StyleHyperMixer(w_dim=w_dim, in_dim=in_dim, hidden_dim=tokens_dim, drop=drop)
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm2 = nn.LayerNorm(in_dim, eps=1e-06)
        self.mlp_style = StyleGLU(w_dim=w_dim, in_dim=in_dim, hidden_dim=channels_dim, drop=drop)
        self.mlp_reduce = GLU(in_dim=in_dim, out_dim=out_dim, drop=drop)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :param w (torch.Tensor): Latent style vector [batch size, 2, latent_dim]
        :return (torch.Tensor): Output tensor of the shape [batch size, tokens, channels]
        """
        x = self.norm1(x)
        x = x + self.drop_path(self.mlp_tokens(x, w[:, 0]))
        x = x + self.drop_path(self.mlp_style(self.norm2(x), w[:, 1]))
        x = self.mlp_reduce(x)
        return x


class Stack(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, None, :].repeat(1, self.n, 1)


def upscale(x):
    return interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


def downscale(x):
    return interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)


class StyleHyperMixerFlyGenerator(nn.Module):
    def __init__(
        self, z_dim=512, w_dim=512, n_map=8, img_resolution=1024, img_channels=3, channel_base=512, drop=0.1, **kwargs
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.n_map = n_map
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.channel_base = channel_base

        self.mapping = nn.Sequential(
            *(
                [GLU(in_dim=z_dim, out_dim=w_dim, drop=0)]
                + [GLU(in_dim=w_dim, out_dim=w_dim, drop=0) for _ in range(n_map - 1)]
                + [Stack(3 * int(np.log2(img_resolution) - 1))]
            )
        )

        self.register_buffer("constant_input", torch.randn((1, channel_base, 4, 4)))

        block_resolutions = 2 ** np.arange(2, np.log2(img_resolution) + 1).astype(int)
        log_n_channels = np.arange(np.log2(channel_base), 4, -1)
        n_channels = np.concatenate(
            (channel_base * np.ones(len(block_resolutions) - len(log_n_channels) + 1), 2**log_n_channels)
        ).astype(int)

        self.synthesis = nn.ModuleList(
            [
                StyleHyperMixerBlock(in_dim, out_dim, w_dim, drop=drop, drop_path=drop)
                for in_dim, out_dim in zip(n_channels[:-1], n_channels[1:])
            ]
        )
        self.to_rgbs = nn.ModuleList(
            [StyleGLU(w_dim, out_dim, out_dim, img_channels, drop, internal=False) for out_dim in n_channels[1:]]
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("StyleHyperMixerFlyGenerator")
        parser.add_argument("--encoder_layers", type=int, default=12)
        parser.add_argument("--data_path", type=str, default="/some/path")
        return parent_parser

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param z (torch.Tensor): Latent style vector [batch size, latent_dim]
        :return (torch.Tensor): Output tensor of the shape [batch size, channels, resolution, resolution]
        """
        ws = self.mapping(z)
        x = self.constant_input.repeat((len(ws), 1, 1, 1))
        img = None
        for i, (block, to_rgb) in enumerate(zip(self.synthesis, self.to_rgbs)):
            w = ws[:, 3 * i : 3 * (i + 1)]
            w12 = w[:, :2]
            w3 = w[:, -1]

            B, _, H, W = x.shape
            x = x.flatten(2).transpose(2, 1)
            x = block(x, w12)
            y = to_rgb(x, w3).transpose(2, 1).reshape(B, -1, H, W)
            img = img + y if img is not None else y

            if W != self.img_resolution:
                x = x.transpose(2, 1).reshape(B, -1, H, W)
                x_img = torch.cat((x, img), dim=1)
                x_img = upscale(x_img)
                x, img = x_img[:, :-3], x_img[:, -3:]

        return img


class HyperMixerFlyDiscriminator(nn.Module):
    def __init__(self, img_resolution=1024, img_channels=3, channel_base=512, drop=0.1) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.channel_base = channel_base

        block_resolutions = 2 ** np.arange(np.log2(img_resolution), 1, -1).astype(int)
        log_n_channels = np.arange(4, np.log2(channel_base))
        n_channels = np.concatenate(
            (2**log_n_channels, channel_base * np.ones(len(block_resolutions) - len(log_n_channels)))
        ).astype(int)

        self.encode = nn.ModuleList(
            [nn.Sequential(Butterfly(img_channels, n_channels[0]), nn.GELU())]
            + [
                HyperMixerBlock(in_dim, out_dim, drop=drop, drop_path=drop)
                for in_dim, out_dim in zip(n_channels[:-1], n_channels[1:])
            ]
        )
        self.predict = GLU(n_channels[-1] * 4 * 4, channel_base, 1, drop=drop, internal=False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img
        for block in self.encode:
            B, _, H, W = x.shape
            x = x.flatten(2).transpose(2, 1)
            x = block(x)
            if W != 4:
                x = x.transpose(2, 1).reshape(B, -1, H, W)
                x = downscale(x)
        logits = self.predict(x.flatten(1))
        return logits


if __name__ == "__main__":

    def print_model_summary(model, input):
        global total_params, already_allocated
        total_params = 0
        already_allocated = 0
        handles = []
        for name, block in model.named_modules():

            def hook(m, i, o, name=name):
                global total_params, already_allocated
                if len(list(m.named_modules())) == 1:
                    class_name = m.__class__.__name__
                    output_shape = (
                        tuple(tuple(oo.shape) if not isinstance(oo, int) else oo for oo in o)
                        if isinstance(o, tuple)
                        else tuple(o.shape)
                    )
                    num_params = sum(p.numel() for p in m.parameters())
                    total_params += num_params
                    allocated = torch.cuda.memory_allocated(0)
                    print(
                        name.ljust(40),
                        class_name.ljust(20),
                        f"{output_shape}".ljust(25),
                        (f"{num_params / 1000:.2f} K" if num_params > 0 else "0").ljust(15),
                        f"{allocated / 1_000_000:.2f} M",
                    )

            handles.append(block.register_forward_hook(hook))

        print()
        print("model summary:")
        print("name".ljust(40), "class".ljust(20), "output shape".ljust(25), "num params".ljust(15), "allocated")
        print("-" * 130)
        out = model(input)
        print("-" * 130)
        print("total".ljust(40), f"".ljust(20), f"{tuple(out.shape)}".ljust(25), f"{total_params/1e6:.2f} M".ljust(15))
        print()

        for handle in handles:
            handle.remove()

    batch_size, z_dim = 2, 512
    img_resolution, img_channels, channel_base = 1024, 3, 512

    G = StyleHyperMixerFlyGenerator(
        z_dim=z_dim, w_dim=z_dim, img_resolution=img_resolution, img_channels=img_channels, channel_base=channel_base
    ).cuda()
    D = HyperMixerFlyDiscriminator(
        img_resolution=img_resolution, img_channels=img_channels, channel_base=channel_base
    ).cuda()
    G.eval(), D.eval()

    with torch.inference_mode():
        print_model_summary(D, torch.randn((batch_size, img_channels, img_resolution, img_resolution), device="cuda"))
        print_model_summary(G, torch.randn(batch_size, z_dim, device="cuda"))

    optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    from time import time

    trials = 20
    print("Benchmarking...")

    G.train(), D.train()
    t = time()
    for _ in range(trials):
        D.requires_grad_(True)
        G.requires_grad_(False)
        D.zero_grad(True)
        img = G(torch.randn(batch_size, z_dim, device="cuda"))
        pred_fake = D(img.detach())
        pred_real = D(torch.randn((batch_size, img_channels, img_resolution, img_resolution), device="cuda"))
        loss_Dgen = torch.nn.functional.softplus(pred_fake).sum()
        loss_Dreal = torch.nn.functional.softplus(-pred_real).sum()
        (loss_Dgen + loss_Dreal).backward()
        optimizer_D.step()

        G.requires_grad_(True)
        D.requires_grad_(False)
        G.zero_grad(True)
        img = G(torch.randn(batch_size, z_dim, device="cuda"))
        pred = D(img)
        loss_G = torch.nn.functional.softplus(-pred).sum()
        loss_G.backward()
        optimizer_G.step()

        torch.cuda.synchronize()

    print("G step")
    print("img", img.min().item(), img.mean().item(), img.max().item())
    print("pred", pred.mean().item())
    print("loss", loss_G.item())
    print()

    print("D step")
    print("pred:", "fake", pred_fake.mean().item(), "real", pred_real.mean().item())
    print("loss:", "fake", loss_Dgen.item(), "real", loss_Dreal.item())
    print()

    print("Train speed")
    print(trials / (time() - t) * batch_size, "img/s")
    print()

    del img, pred_fake, pred_real, pred, loss_Dgen, loss_Dreal, loss_G
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    D.zero_grad(True)
    D.requires_grad_(False)
    G.zero_grad(True)
    G.requires_grad_(False)
    G.eval(), D.eval()

    with torch.inference_mode():
        t = time()
        for _ in range(trials):
            img = G(torch.randn(batch_size, z_dim, device="cuda"))
            torch.cuda.synchronize()
        print("Inference speed")
        print(trials / (time() - t) * batch_size, "img/s")
