from functools import partial

from torch import nn


class Perceptor(nn.Module):
    def __init__(self, content_layers, style_layers) -> None:
        super().__init__()
        self.content_layers, self.style_layers = content_layers, style_layers

    def forward(self, inputs, targets):
        raise NotImplementedError()


from .vgg_pgg import PGGPerceptor


def load_perceptor(name: str) -> Perceptor:
    if name.startswith("pgg"):
        return partial(PGGPerceptor, model_name=name.replace("pgg-", ""))
