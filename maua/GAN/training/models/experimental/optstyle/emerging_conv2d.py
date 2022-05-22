import numpy as np
import pyximport
import torch
from torch.nn.functional import conv2d, l1_loss, mse_loss, pad
from tqdm import trange

from maua.GAN.wrappers import get_generator_class

pyximport.install(inplace=True)

# from . import inverse_op_naive as inverse_op
from . import inverse_op_cython as inverse_op


def rmse(x, y):
    return torch.sqrt(torch.mean(torch.pow(x - y, 2)))


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)
    mask = torch.ones([n_in, n_out])
    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i + 1 :, i * k : (i + 1) * k] = 0
            if zerodiagonal:
                mask[i : i + 1, i * k : (i + 1) * k] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[(i + 1) * k :, i : i + 1] = 0
            if zerodiagonal:
                mask[i * k : (i + 1) * k :, i : i + 1] = 0
    return mask


def get_conv_square_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    """
    Function to get autoregressive convolution with square shape.
    """
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = torch.ones([h, w, n_in, n_out])
    mask[:l, :, :, :] = 0
    mask[:, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask


def get_conv_weight(filter_shape, stable_init=True, unit_testing=False):
    weight = torch.randn(*filter_shape) * 0.002
    center = (filter_shape[0] - 1) // 2
    if stable_init or unit_testing:
        weight[center, center, :, :] += torch.eye(filter_shape[3])
    return weight


def inverse_conv(z, w, is_upper, dilation):
    z, w = z.permute(0, 2, 3, 1), w.permute(2, 3, 1, 0)

    z_np = z.cpu().numpy()
    w_np = w.cpu().numpy()

    center = (w_np.shape[0] - 1) // 2
    diagonal = np.diag(w_np[center, center, :, :])
    alpha = 1.0 / np.min(np.abs(diagonal))
    alpha = max(1.0, alpha)

    w_np *= alpha
    x_np = inverse_op.inverse_conv(z_np, w_np, int(is_upper), dilation)
    x_np *= alpha

    return torch.from_numpy(x_np).permute(0, 3, 1, 2).to(z)


class EmergingConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()

        assert stride == 1
        assert (kernel_size - 1) % 2 == 0

        self.kernel_size = kernel_size
        self.center = (kernel_size - 1) // 2
        self.stride = stride
        self.dilation = dilation

        filter_shape = [kernel_size, kernel_size, in_channels, out_channels]
        self.w1 = torch.nn.Parameter(get_conv_weight(filter_shape).permute(3, 2, 0, 1))
        self.w2 = torch.nn.Parameter(get_conv_weight(filter_shape).permute(3, 2, 0, 1))
        self.b = torch.nn.Parameter(torch.zeros((1, out_channels, 1, 1)))
        self.register_buffer("Lmask", get_conv_square_ar_mask(*filter_shape).permute(3, 2, 0, 1))
        self.register_buffer("Umask", self.Lmask.flip((0, 1, 2, 3)))

    def forward(self, z, reverse=False):
        w1, w2 = self.Lmask * self.w1, self.Umask * self.w2

        if reverse:
            x = z - self.b
            x = inverse_conv(x, w2, is_upper=1, dilation=self.dilation)
            x = inverse_conv(x, w1, is_upper=0, dilation=self.dilation)
            return x

        else:
            # Smaller versions of w1, w2.
            w1_s = w1[..., self.center :, self.center :]
            w2_s = w2[..., : -self.center, : -self.center]

            padding = self.center * self.dilation

            z = pad(z, (0, padding, 0, padding))
            z = conv2d(z, w1_s, stride=self.stride, dilation=self.dilation)

            z = pad(z, (padding, 0, padding, 0))
            z = conv2d(z, w2_s, stride=self.stride, dilation=self.dilation)

            z = z + self.b

            return z


class InvertibleLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super(InvertibleLeakyReLU, self).__init__()
        self.negative_slope = torch.nn.Parameter(torch.tensor(negative_slope))

    def forward(self, input, reverse=False):
        if reverse:
            return torch.where(input >= 0.0, input, input * (1 / self.negative_slope))
        else:
            return torch.where(input >= 0.0, input, input * (self.negative_slope))


class InvertibleSequential(torch.nn.Module):
    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input, reverse=False):
        x = input
        for lay in self.layers:
            x = lay(x, reverse=reverse)
        return x


class StyleGAN3Partial:
    def __init__(self, G_synth):
        self.layers = list(G_synth.children())
        self.output_scale = G_synth.output_scale

    def forward_to_depth(self, ws, depth):
        c = 0
        ws = ws.float().unbind(dim=1)
        x = self.layers[0](ws[0])
        c += 1

        for layer, w in zip(self.layers[1:], ws[1:]):
            if c > depth:
                return x

            x = layer(x, w)

            c += 1

        if self.output_scale != 1:
            x = x * self.output_scale

        return x.float()


if __name__ == "__main__":
    n_iters = 500
    batch_size = 128
    train_size = 128

    G = get_generator_class("stylegan3")(
        model_file="/home/hans/modelzoo/stylegan3/00003-stylegan3-r-diffuse-gpus1-batch4-gamma6.6/network-snapshot-000940.pkl",
        output_size=(train_size, train_size),
        strategy="stretch",
        layer=0,
    ).cuda()

    layer_shapes = []

    def get_shape(mod, inp, out):
        layer_shapes.append((inp[0].shape, out.shape))
        return out

    teacher_layers = list(G.synthesizer.G_synth.children())[1:]
    hooks = [layer.register_forward_hook(get_shape) for layer in teacher_layers]
    G(torch.randn(batch_size, 512, device="cuda"))
    for hook in hooks:
        hook.remove()

    M = G.mapper
    G = StyleGAN3Partial(G.synthesizer.G_synth)

    invertible_layers = []
    for d, ((in_shape, out_shape), layer) in enumerate(zip(layer_shapes, teacher_layers)):
        d += 1
        print(d, in_shape, out_shape)

        inverter = InvertibleSequential(
            EmergingConv2d(in_shape[1], in_shape[1], kernel_size=5),
            InvertibleLeakyReLU(),
            EmergingConv2d(in_shape[1], out_shape[1], kernel_size=5),
            InvertibleLeakyReLU(),
        ).cuda()

        optim = torch.optim.Adam(inverter.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_iters, eta_min=1e-6)

        with trange(n_iters) as progress:
            for it in progress:

                lat = torch.randn((batch_size, 512), device="cuda")
                w = M(lat)
                if d > 0:
                    x = G.forward_to_depth(w, d)
                else:
                    x = w
                z_teacher = layer(x, w[:, d])
                z_student = inverter(x)

                optim.zero_grad()
                loss = mse_loss(z_student, z_teacher)
                loss.backward()
                optim.step()
                scheduler.step()

                if it % 25 == 0:
                    abspdiff = (z_student - z_teacher).div(torch.maximum(z_student, z_teacher)).abs().mul(100)
                    progress.write(
                        f"{loss.item():.2f}".ljust(10)
                        + f"{torch.min(abspdiff).item():.2f}%".ljust(10)
                        + f"{torch.mean(abspdiff).item():.2f}%".ljust(10)
                        + f"{torch.max(abspdiff).item():.2f}%".ljust(10)
                        + f"{(z_student.min() - z_teacher.min()).item():.4f}".ljust(10)
                        + f"{(z_student.mean() - z_teacher.mean()).item():.4f}".ljust(10)
                        + f"{(z_student.max() - z_teacher.max()).item():.4f}"
                    )
                    # with torch.no_grad():
                    #     x = torch.randn(in_shape, device="cuda")[:4]
                    #     z = inverter(x)
                    #     x_recon = inverter(z, reverse=True)
                    #     z_recon = inverter(x_recon)
                    #     progress.write(f"x recon rmse {rmse(x, x_recon).item()}")
                    #     progress.write(f"z recon rmse {rmse(z, z_recon).item()}")

        invertible_layers.append(inverter)

    full_inverter = torch.nn.Sequential(invertible_layers)
    print(full_inverter)
    torch.save(full_inverter, "inverter.pt")

    with torch.no_grad():
        x = torch.randn(in_shape, device="cuda")[:4]
        z = inverter(x)
        x_recon = inverter(z, reverse=True)
        z_recon = inverter(x_recon)
        print("x recon rmse", rmse(x, x_recon).item())
        print("z recon rmse", rmse(z, z_recon).item())

    exit(0)

    shape = [128, 64, 32, 48]
    layer = EmergingConv2d(64, 64, kernel_size=27).cuda()
    w, b = torch.randn((64, 64, 3, 3)).cuda(), torch.randn((1, 64, 1, 1)).cuda()
    optim = torch.optim.Adam(layer.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_iters, eta_min=1e-6)

    for it in range(n_iters):

        optim.zero_grad()
        x = torch.randn(shape, device="cuda")
        z = layer(x)
        zr = conv2d(x, w, padding=1) + b
        mse = mse_loss(z, zr)
        l1 = 10 * l1_loss(z, zr)
        loss = mse + l1
        loss.backward()
        optim.step()
        scheduler.step()

        if it % 100 == 0:
            abspdiff = (z - zr).div(torch.maximum(z, zr)).abs().mul(100)
            print(
                f"{mse.item():.4f}".ljust(10),
                f"{l1.item():.4f}".ljust(10),
                f"{torch.quantile(abspdiff,0.01).item():.4f}".ljust(10),
                f"{torch.quantile(abspdiff,0.05).item():.4f}".ljust(10),
                f"{torch.quantile(abspdiff,0.25).item():.4f}".ljust(10),
                f"{torch.quantile(abspdiff,0.50).item():.4f}".ljust(10),
                f"{torch.quantile(abspdiff,0.75).item():.4f}".ljust(10),
                f"{torch.quantile(abspdiff,0.95).item():.4f}".ljust(10),
                f"{torch.quantile(abspdiff,0.99).item():.4f}".ljust(10),
            )

    with torch.no_grad():
        x = torch.randn(shape)
        z = layer(x)
        layer.reverse = True
        x_recon = layer(z)
        layer.reverse = False
        z_recon = layer(x_recon)

        print("x recon rmse", rmse(x, x_recon).item())
        print("z recon rmse", rmse(z, z_recon).item())

        exit(0)

        layers = [EmergingConv2d(shape[1], shape[1]) for _ in range(10)]
        z = torch.randn(shape)
        for layer in layers:
            z = layer(z)
        for layer in reversed(layers):
            z = layer(z, reverse=True)
        print("multi layer recon rmse", rmse(x, z).item())
        print()
