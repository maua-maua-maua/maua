from functools import partial

import torch
from torch import nn

from maua.loss import feature_loss, gram_matrix


class Perceptor(nn.Module):
    def __init__(self, content_strength, content_layers, style_strength, style_layers) -> None:
        super().__init__()
        self.content_layers, self.style_layers = content_layers, style_layers
        self.content_strength, self.style_strength = content_strength, style_strength

        self.embeddings = [None for _ in content_layers + style_layers]
        self.targets = None
        self.loss = 0

    def register_layer_hooks(self):
        c = -1  # magical init value so that second for doesn't off-by-one error when there are no content_layers

        for c, layer in enumerate(self.content_layers):

            def content_hook(module, input, output, l=c):
                embedding = output.squeeze().flatten(1)
                if self.targets is None:
                    self.embeddings[l] = embedding
                else:
                    self.loss += self.content_strength * feature_loss(embedding, self.targets[l])

            getattr(self.net, str(layer)).register_forward_hook(content_hook)

        for s, layer in enumerate(self.style_layers):

            def style_hook(module, input, output, l=c + 1 + s):
                embedding = gram_matrix(output)
                if self.targets is None:
                    self.embeddings[l] = embedding
                else:
                    self.loss += self.style_strength * feature_loss(embedding, self.targets[l])

            getattr(self.net, str(layer)).register_forward_hook(style_hook)

    def get_target_embeddings(self, contents=None, styles=None, content_weights=None, style_weights=None):
        if isinstance(contents, torch.Tensor):
            contents = [contents]

        content_embeddings = None
        if contents is not None:
            if content_weights is None:
                content_weights = torch.ones(len(contents))
            content_weights /= content_weights.sum()

            for content, content_weight in zip(contents, content_weights):
                embs = self.forward(content)[: len(self.content_layers)]
                wembs = torch._foreach_mul(embs, content_weight)
                if content_embeddings is None:
                    content_embeddings = wembs
                else:
                    content_embeddings = torch._foreach_add(content_embeddings, wembs)

        style_embeddings = None
        if styles is not None:
            if style_weights is None:
                style_weights = torch.ones(len(styles))
            style_weights /= style_weights.sum()

            for style, style_weight in zip(styles, style_weights):
                embs = self.forward(style)[len(self.content_layers) :]
                wembs = torch._foreach_mul(embs, style_weight)
                if style_embeddings is None:
                    style_embeddings = wembs
                else:
                    style_embeddings = torch._foreach_add(style_embeddings, wembs)

        if content_embeddings is None:
            return style_embeddings
        if style_embeddings is None:
            return content_embeddings
        return (*content_embeddings, *style_embeddings)

    def forward(self, x) -> list[torch.Tensor]:
        self.net(self.preprocess(x))
        return self.embeddings

    def get_loss(self, x, targets):
        assert len(targets) == len(self.embeddings), (
            f"The target embeddings don't match this perceptor's embeddings: {len(targets)}. Expected: {len(self.embeddings)}"
        )
        self.loss = 0
        self.targets = targets
        self.forward(x)
        self.targets = None
        return self.loss


from maua.perceptors.vgg_kbc import KBCPerceptor
from maua.perceptors.vgg_pgg import PGGPerceptor


def load_perceptor(name: str) -> Perceptor:
    if name.startswith("pgg"):
        return partial(PGGPerceptor, model_name=name.replace("pgg-", ""))
    if name.startswith("kbc"):
        return KBCPerceptor
