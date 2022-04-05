import numpy as np
import torch
from involution import Involution2d
from torch.nn import GELU, LayerNorm, Module, Sequential, UpsamplingBilinear2d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Norm") != -1:
        torch.nn.init.normal_(m.weight.data, 0.4, 0.02)  # 0.4 chosen such that G output is rougly in (-3, 3) on init
        torch.nn.init.constant_(m.bias.data, 0)


class DeepInvolutionalGenerator(Module):
    def __init__(self, image_size=64, nz=100, ngf=64, nc=3, **kwargs):
        super().__init__()

        nb = round(np.log2(image_size)) - 1
        nfs = [nz] + list(reversed([min(ngf * 2**i, ngf * 8) for i in range(nb)])) + [nc]
        res = 1

        blocks = []
        for b, (nf_prev, nf_inter, nf_next) in enumerate(zip(nfs[:-1], [nfs[1]] + nfs[1:-1], nfs[1:])):
            blocks += [
                Involution2d(nf_prev, nf_inter, sigma_mapping=Sequential(LayerNorm([nf_inter, res, res]), GELU())),
                LayerNorm([nf_inter, res, res]),
                GELU(),
                UpsamplingBilinear2d(scale_factor=2),
                Involution2d(
                    nf_inter, nf_next, sigma_mapping=Sequential(LayerNorm([nf_next, res * 2, res * 2]), GELU())
                ),
            ]
            if b < nb:
                blocks += [LayerNorm([nf_next, res * 2, res * 2]), GELU()]
            res *= 2

        self.main = Sequential(*blocks)
        self.main.apply(weights_init)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DeepInvolutionalGenerator")
        parser.add_argument("--nz", type=int, default=100, help="Size of the latent space")
        parser.add_argument("--ngf", type=int, default=64, help="Base number of filters in the generator")
        return parent_parser

    def forward(self, input):
        return self.main(input[..., None, None])


class DeepInvolutionalDiscriminator(Module):
    def __init__(self, image_size=64, nc=3, ndf=64, **kwargs):
        super().__init__()

        nb = round(np.log2(image_size)) - 1
        nfs = [nc] + list([min(ndf * 2**i, ndf * 8) for i in range(nb)]) + [1]
        res = image_size

        blocks = []
        for b, (nf_prev, nf_inter, nf_next) in enumerate(zip(nfs[:-1], [nfs[1]] + nfs[1:-1], nfs[1:])):
            blocks += [
                Involution2d(nf_prev, nf_inter, sigma_mapping=Sequential(LayerNorm([nf_inter, res, res]), GELU())),
                LayerNorm([nf_inter, res, res]),
                GELU(),
                Involution2d(
                    nf_inter,
                    nf_next,
                    stride=2,
                    sigma_mapping=Sequential(LayerNorm([nf_next, res // 2, res // 2]), GELU()),
                ),
            ]
            if b < nb:
                blocks += [LayerNorm([nf_next, res // 2, res // 2]), GELU()]
            res //= 2

        self.main = Sequential(*blocks)
        self.main.apply(weights_init)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DeepInvolutionalDiscriminator")
        parser.add_argument("--ndf", type=int, default=64, help="Base number of filters in the discriminator")
        return parent_parser

    def forward(self, input):
        return self.main(input).squeeze()


if __name__ == "__main__":
    B = 32
    with torch.inference_mode():
        vals = [[], []]
        for _ in range(100):
            img = DeepInvolutionalGenerator(image_size=64).cuda()(torch.randn(B, 100).cuda())
            vals[0].append(img.min().item())
            vals[1].append(img.max().item())
        print("min", np.mean(vals[0]), "+/-", np.std(vals[0]))
        print("max", np.mean(vals[1]), "+/-", np.std(vals[1]))

    G = DeepInvolutionalGenerator(image_size=64).cuda()
    D = DeepInvolutionalDiscriminator(image_size=64).cuda()
    z = torch.randn(B, 100).cuda()
    z.requires_grad_()
    img = G(z)
    pred = D(img)
    loss_G = torch.nn.functional.softplus(-pred).sum()
    loss_G.backward()
    print(z.grad.norm().item())
