from collections import OrderedDict
from os import path

import gdown
import torch
import torch.nn as nn
from nestedtensor import nested_tensor
from torch.utils.model_zoo import load_url

from maua.ops.loss import feature_loss, gram_matrix
from maua.perceptors import Perceptor
from maua.utility import download


class PGGPerceptor(Perceptor):
    """VGG networks ported by ProGamerGov from Caffe for his amazing style transfer implementation neural-style-pt"""

    def __init__(
        self,
        content_layers=None,
        style_layers=None,
        content_strength=1,
        style_strength=1,
        model_name="vgg19",
        pooling="max",
    ):
        if content_layers is None:
            content_layers = default_layers[model_name]["content"]
        if style_layers is None:
            style_layers = default_layers[model_name]["style"]

        super().__init__(content_layers, style_layers)

        net = select_model(model_name.lower(), pooling)
        self.net = nn.Sequential(
            *list(net.children())[: max(content_layers + style_layers) + 1]  # remove unnecessary layers
        )

        # convert to BGR and scale to range Caffe VGGs expect
        self.preprocess = (
            lambda x: 255 * x[:, [2, 1, 0]]
            - torch.tensor([[103.939, 116.779, 123.68]], device=x.device)[..., None, None]
        )

        self.embeddings = [None for _ in content_layers + style_layers]
        self.targets = None
        self.loss = 0

        for c, layer in enumerate(content_layers):

            def content_hook(module, input, output, l=c):
                embedding = output.squeeze().flatten(1)
                if self.targets is None:
                    self.embeddings[l] = embedding
                else:
                    self.loss += content_strength * feature_loss(embedding, self.targets[l])

            getattr(net, str(layer)).register_forward_hook(content_hook)

        for s, layer in enumerate(style_layers):

            def style_hook(module, input, output, l=c + 1 + s):
                embedding = gram_matrix(output)
                if self.targets is None:
                    self.embeddings[l] = embedding
                else:
                    self.loss += style_strength * feature_loss(embedding, self.targets[l])

            getattr(net, str(layer)).register_forward_hook(style_hook)

    def get_target_embeddings(self, contents=None, styles=None, content_weights=None, style_weights=None):
        if isinstance(contents, torch.Tensor):
            contents = [contents]

        content_embeddings = None
        if contents is not None:
            if content_weights is None:
                content_weights = torch.ones(len(contents))
            content_weights /= content_weights.sum()

            for content, content_weight in zip(contents, content_weights):
                if content_embeddings is None:
                    content_embeddings = content_weight * self.forward(content)[: len(self.content_layers)]
                else:
                    content_embeddings += content_weight * self.forward(content)[: len(self.content_layers)]

        style_embeddings = None
        if styles is not None:
            if style_weights is None:
                style_weights = torch.ones(len(styles))
            style_weights /= style_weights.sum()

            for style, style_weight in zip(styles, style_weights):
                if style_embeddings is None:
                    style_embeddings = style_weight * self.forward(style)[len(self.content_layers) :]
                else:
                    style_embeddings += style_weight * self.forward(style)[len(self.content_layers) :]

        if content_embeddings is None:
            return style_embeddings
        if style_embeddings is None:
            return content_embeddings
        return torch.cat((content_embeddings, style_embeddings))

    def forward(self, x):
        self.net(self.preprocess(x))
        return nested_tensor(self.embeddings, device=x.device)

    def get_loss(self, x, targets):
        assert len(targets) == len(
            self.embeddings
        ), f"The target embeddings don't match this perceptor's embeddings: {len(targets)}. Expected: {len(self.embeddings)}"
        self.loss = 0
        self.targets = targets
        self.forward(x)
        self.targets = None
        return self.loss


default_layers = {
    "prune": {"content": [22], "style": [3, 8, 15, 22, 29]},
    "nyud": {"content": [22], "style": [3, 8, 15, 22, 29]},
    "fcn32s": {"content": [22], "style": [3, 8, 15, 22, 29]},
    "sod": {"content": [22], "style": [3, 8, 15, 22, 29]},
    "vgg16": {"content": [22], "style": [3, 8, 15, 22, 29]},
    "vgg19": {"content": [26], "style": [3, 8, 17, 26, 35]},
    "nin": {"content": [19], "style": [5, 12, 19, 27]},
}
channel_list = {
    "VGG-16p": [24, 22, "P", 41, 51, "P", 108, 89, 111, "P", 184, 276, 228, "P", 512, 512, 512, "P"],
    "VGG-16": [64, 64, "P", 128, 128, "P", 256, 256, 256, "P", 512, 512, 512, "P", 512, 512, 512, "P"],
    "VGG-19": [64, 64, "P", 128, 128, "P", 256, 256, 256, 256, "P", 512, 512, 512, 512, "P", 512, 512, 512, 512, "P"],
}


def select_model(model_name, pooling):
    if "prun" in model_name:
        model_file = "modelzoo/vgg16-prune.pth"
        if not path.exists(model_file):
            gdown.download("https://drive.google.com/uc?id=1aaNqJ5D2A-vev3IZFv6dSkovuA3XwYsq", model_file)
        cnn = VGG_PRUNED(build_sequential(channel_list["VGG-16p"], pooling))

    elif "nyud" in model_name:
        model_file = "modelzoo/nyud-fcn32s-color-heavy.pth"
        if not path.exists(model_file):
            gdown.download("https://drive.google.com/uc?id=1MKj6Dntzh7t45PxM4I0ixWaQtisAg9hy", model_file)
        cnn = VGG_FCN32S(build_sequential(channel_list["VGG-16"], pooling))

    elif "fcn32s" in model_name:
        model_file = "modelzoo/fcn32s-heavy-pascal.pth"
        if not path.exists(model_file):
            gdown.download("https://drive.google.com/uc?id=1bcAnvfMuuEbJqjaVWIUCD9HUgD1fvxI_", model_file)
        cnn = VGG_FCN32S(build_sequential(channel_list["VGG-16"], pooling))

    elif "sod" in model_name:
        model_file = "modelzoo/vgg16-sod.pth"
        if not path.exists(model_file):
            gdown.download("https://drive.google.com/uc?id=1EU-F9ugeIeTO9ay4PinzsBXgEuCYBu0Z", model_file)
        cnn = VGG_SOD(build_sequential(channel_list["VGG-16"], pooling))

    elif "vgg16" in model_name:
        model_file = "modelzoo/vgg16.pth"
        if not path.exists(model_file):
            sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth")
            map = {
                "classifier.1.weight": "classifier.0.weight",
                "classifier.1.bias": "classifier.0.bias",
                "classifier.4.weight": "classifier.3.weight",
                "classifier.4.bias": "classifier.3.bias",
            }
            sd = OrderedDict([(map[k] if k in map else k, v) for k, v in sd.items()])
            torch.save(sd, model_file)
        cnn = VGG(build_sequential(channel_list["VGG-16"], pooling))

    elif "vgg19" in model_name:
        model_file = "modelzoo/vgg19.pth"
        if not path.exists(model_file):
            sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth")
            map = {
                "classifier.1.weight": "classifier.0.weight",
                "classifier.1.bias": "classifier.0.bias",
                "classifier.4.weight": "classifier.3.weight",
                "classifier.4.bias": "classifier.3.bias",
            }
            sd = OrderedDict([(map[k] if k in map else k, v) for k, v in sd.items()])
            torch.save(sd, model_file)
        cnn = VGG(build_sequential(channel_list["VGG-19"], pooling))

    elif "nin" in model_name:
        model_file = "modelzoo/nin.pth"
        if not path.exists(model_file):
            download("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", model_file)
        cnn = NIN(pooling)

    else:
        raise ValueError("Model architecture not recognized.")

    cnn.load_state_dict(torch.load(model_file))

    for param in cnn.parameters():
        param.requires_grad = False

    return cnn.features


def build_sequential(channel_list, pooling):
    layers = []
    in_channels = 3
    for c in channel_list:
        if c == "P":
            layers += [
                nn.MaxPool2d(kernel_size=2, stride=2) if pooling == "max" else nn.AvgPool2d(kernel_size=2, stride=2)
            ]
        else:
            layers += [nn.Conv2d(in_channels, c, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


class VGG_SOD(nn.Module):
    def __init__(self, features, num_classes=100):
        super(VGG_SOD, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 100),
        )


class VGG_FCN32S(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_FCN32S, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, (7, 7)),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, (1, 1)),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )


class VGG_PRUNED(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_PRUNED, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )


class NIN(nn.Module):
    def __init__(self, pooling):
        super(NIN, self).__init__()
        if pooling == "max":
            pool2d = nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        elif pooling == "avg":
            pool2d = nn.AvgPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), (4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, (1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, (1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Dropout(0.5),
            nn.Conv2d(384, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1000, (1, 1)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((6, 6), (1, 1), (0, 0), ceil_mode=True),
            nn.Softmax(),
        )
