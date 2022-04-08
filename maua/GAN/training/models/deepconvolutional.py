import numpy as np
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class DeepConvolutionalGenerator(torch.nn.Module):
    def __init__(self, image_size=64, z_dim=100, ngf=64, img_channels=3, **kwargs):
        super().__init__()

        nb = round(np.log2(image_size)) - 2
        nfs = [z_dim] + list(reversed([min(ngf * 2**i, ngf * 8) for i in range(nb)])) + [img_channels]

        blocks = []
        for b, (nf_prev, nf_next) in enumerate(zip(nfs[:-1], nfs[1:])):
            blocks += [
                torch.nn.ConvTranspose2d(
                    nf_prev, nf_next, kernel_size=4, stride=1 if b == 0 else 2, padding=0 if b == 0 else 1, bias=False
                )
            ]
            if b < nb:
                blocks += [torch.nn.BatchNorm2d(nf_next), torch.nn.LeakyReLU(0.2, inplace=True)]
            else:
                blocks += [torch.nn.Tanh()]

        self.main = torch.nn.Sequential(*blocks)
        self.main.apply(weights_init)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DeepConvolutionalGenerator")
        parser.add_argument("--z_dim", type=int, default=100, help="Size of the latent space")
        parser.add_argument("--ngf", type=int, default=64, help="Base number of filters in the generator")
        return parent_parser

    def forward(self, input):
        return self.main(input[..., None, None])


class DeepConvolutionalDiscriminator(torch.nn.Module):
    override_args = {"logits": True}

    def __init__(self, image_size=64, img_channels=3, ndf=64, **kwargs):
        super().__init__()

        nb = round(np.log2(image_size)) - 2
        nfs = [img_channels] + list([min(ndf * 2**i, ndf * 8) for i in range(nb)]) + [1]

        blocks = []
        for b, (nf_prev, nf_next) in enumerate(zip(nfs[:-1], nfs[1:])):
            blocks += [
                torch.nn.Conv2d(
                    nf_prev, nf_next, kernel_size=4, stride=1 if b == nb else 2, padding=0 if b == nb else 1, bias=False
                )
            ]
            if b < nb:
                blocks += [torch.nn.BatchNorm2d(nf_next), torch.nn.LeakyReLU(0.2, inplace=True)]

        self.main = torch.nn.Sequential(*blocks)
        self.main.apply(weights_init)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DeepConvolutionalDiscriminator")
        parser.add_argument("--ndf", type=int, default=64, help="Base number of filters in the discriminator")
        return parent_parser

    def forward(self, input):
        return self.main(input).squeeze()


if __name__ == "__main__":
    G = DeepConvolutionalGenerator(image_size=128, z_dim=64, ngf=128)
    print(G)
    print()

    D = DeepConvolutionalDiscriminator(image_size=128, img_channels=3, ndf=128)
    print(D)
    print()

    z = torch.randn((32, 64))
    img = G(z)
    print(img.shape)

    pred = D(img)
    print(pred.shape)
