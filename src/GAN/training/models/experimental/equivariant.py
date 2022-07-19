from typing import List, Optional, Tuple

import numpy as np
import torch
from escnn import gspaces, nn


class ExtractRotation(nn.EquivariantModule):
    """
    Extract the regular representation corresponding to a single rotation from a vector field
    """

    def __init__(self, gspace: gspaces.GSpace, channels: int, irreps: List):
        assert isinstance(gspace, gspaces.GSpace)
        super(ExtractRotation, self).__init__()

        self.space = gspace
        self.G = gspace.fibergroup
        self.rho = self.G.spectral_regular_representation(*irreps)

        self.in_type = nn.FieldType(self.space, [self.rho] * channels)
        self.out_type = nn.FieldType(self.space, [self.space.trivial_repr] * channels)

        kernel = []
        for irr in irreps:
            irr = self.G.irrep(*irr)
            c = int(irr.size // irr.sum_of_squares_constituents)
            k = irr(self.G.identity)[:, :c] * np.sqrt(irr.size)
            kernel.append(k.T.reshape(-1))
        kernel = np.concatenate(kernel)
        assert kernel.shape[0] == self.rho.size
        kernel = kernel / np.linalg.norm(kernel)
        kernel = kernel.reshape(-1, 1)
        self.register_buffer("kernel", torch.tensor(kernel, dtype=torch.get_default_dtype()))

    def forward(self, input: nn.GeometricTensor, rotation: Optional[float]) -> nn.GeometricTensor:
        assert input.type == self.in_type
        shape = input.shape
        x_hat = input.tensor.view(shape[0], len(self.in_type), self.rho.size, *shape[2:])

        if rotation is None:
            g = self.G.element((np.random.randint(0, 2), np.random.random() * 2 * np.pi), param="radians")
        else:
            g = self.G.element((0, rotation / 180 * np.pi), param="radians")
        A = torch.cat([torch.tensor(self.rho(g)).to(input.tensor) @ self.kernel], dim=1).T

        x = torch.einsum("bcf...,gf->bcg...", x_hat, A)  # TODO batched rotations
        y = x[:, :, 0]

        return nn.GeometricTensor(y, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size
        return (input_shape[0], self.out_type.size, *input_shape[2:])


class SteerableGenerator(torch.nn.Module):
    def __init__(self, latent_dim=128, n_mlp=4, n_channels=3, n_filters=64, maximum_frequency=6):
        super(SteerableGenerator, self).__init__()

        # Mapping Network
        self.mapping = torch.nn.Sequential(
            *[torch.nn.Linear(latent_dim, latent_dim), torch.nn.ELU(inplace=True)] * n_mlp
        )

        # the model is equivariant under arbitrary rotations and flips
        self.gspace = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.gspace, [self.gspace.trivial_repr] * latent_dim)
        self.input_type = in_type
        irreps = [(1, k) for k in range(maximum_frequency + 1)]
        main_type = nn.FieldType(self.gspace, [self.gspace.irrep(*id) for id in irreps])

        blocks = []
        for c, channels in enumerate(
            [n_filters * 3, n_filters * 3, n_filters * 2, n_filters * 2, n_filters, n_filters, n_channels]
        ):
            out_type = nn.FieldType(self.gspace, channels * [main_type.representation])
            blocks.append(
                nn.SequentialModule(
                    nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
                    nn.NormBatchNorm(out_type),
                    nn.NormNonLinearity(out_type),
                )
            )
            if c % 2 == 1:
                blocks.append(nn.R2Upsampling(out_type, scale_factor=2))
            in_type = out_type

        spectral_type = nn.FieldType(
            self.gspace, channels * [self.gspace.fibergroup.spectral_regular_representation(*irreps)]
        )
        blocks.append(nn.R2Conv(in_type, spectral_type, kernel_size=3, padding=1))

        self.synthesis = nn.SequentialModule(*blocks)

        self.extract_rotation = ExtractRotation(self.gspace, channels, irreps)

    def forward(self, z: torch.Tensor, r: Optional[float] = None):
        w = self.mapping(z)
        w = w[..., None, None].tile(1, 1, 4, 4)
        x = nn.GeometricTensor(w, self.input_type)
        x = self.synthesis(x)
        x = self.extract_rotation(x, r)
        return x.tensor


class SteerableDiscriminator(torch.nn.Module):
    def __init__(self, image_size=32, n_channels=3, n_filters=64, maximum_frequency=6):
        super(SteerableDiscriminator, self).__init__()

        # the model is equivariant under arbitrary rotations and flips
        self.gspace = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.gspace, [self.gspace.trivial_repr] * n_channels)
        self.input_type = in_type
        main_type = nn.FieldType(self.gspace, [self.gspace.irrep(1, k) for k in range(maximum_frequency + 1)])

        blocks = [nn.MaskModule(in_type, image_size, margin=1)]
        for c, channels in enumerate(
            [n_filters, n_filters, n_filters * 2, n_filters * 2, n_filters * 3, n_filters * 3]
        ):
            out_type = nn.FieldType(self.gspace, channels * [main_type.representation])
            blocks.append(
                nn.SequentialModule(
                    nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
                    nn.NormBatchNorm(out_type),
                    nn.NormNonLinearity(out_type),
                )
            )
            if c % 2 == 1:
                blocks.append(nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=2))
            in_type = out_type
        blocks.append(nn.NormPool(out_type))

        self.main = nn.SequentialModule(*blocks)

        self.fc = torch.nn.Sequential(  # 4x4
            torch.nn.Conv2d(blocks[-1].out_type.size, n_filters, kernel_size=3, bias=False),  # 2x2
            torch.nn.BatchNorm2d(n_filters),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(n_filters, n_filters, kernel_size=2, bias=False),  # 1x1
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(n_filters),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(n_filters, n_filters),
            torch.nn.BatchNorm1d(n_filters),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(n_filters, 1),
        )

    def forward(self, img: torch.Tensor):
        x = nn.GeometricTensor(img, self.input_type)
        x = self.main(x)
        x = x.tensor
        x = self.fc(x)
        return x


if __name__ == "__main__":
    with torch.no_grad():
        batch_size, image_size, latent_dim, n_mlp, n_channels, n_filters = 16, 32, 128, 4, 3, 64
        G = SteerableGenerator(latent_dim=latent_dim, n_mlp=n_mlp, n_channels=n_channels, n_filters=n_filters).cuda()
        D = SteerableDiscriminator(image_size=image_size, n_channels=n_channels, n_filters=n_filters).cuda()

        z = torch.randn(batch_size, latent_dim).cuda()
        x = G(z)
        print("G out:", x.shape, x.min(), x.mean(), x.max())
        y = D(x)
        print("D out:", y.shape, y.min(), y.mean(), y.max())

        # the outputs should be (about) the same for all transformations the model is invariant to
        y_fv = D(x.flip(dims=[3]))
        y_fh = D(x.flip(dims=[2]))
        y90 = D(x.rot90(1, (2, 3)))
        y90_fh = D(x.flip(dims=[2]).rot90(1, (2, 3)))
        print("\nTESTING DISCRIMINATOR INVARIANCE:")
        print("REFLECTIONS along the VERTICAL axis:   " + ("YES" if torch.allclose(y, y_fv, atol=1e-6) else "NO"))
        print("REFLECTIONS along the HORIZONTAL axis: " + ("YES" if torch.allclose(y, y_fh, atol=1e-6) else "NO"))
        print("90 degrees ROTATIONS:                  " + ("YES" if torch.allclose(y, y90, atol=1e-6) else "NO"))
        print("REFLECTIONS along the 45 degrees axis: " + ("YES" if torch.allclose(y, y90_fh, atol=1e-6) else "NO"))

        import torchvision as tv

        z = torch.randn(1, latent_dim).cuda()
        video = torch.cat([G(z, r) for r in range(0, 360, 2)], dim=0).permute(0, 2, 3, 1)
        video -= video.min()
        video /= video.max()
        video *= 255
        tv.io.write_video("/tmp/generator360.mp4", video.cpu(), fps=36)
