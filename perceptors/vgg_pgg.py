import copy
import urllib.request
from collections import OrderedDict
from os import path

import gdown
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url


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


def build_sequential(channel_list, pooling):
    layers = []
    in_channels = 3
    if pooling == "max":
        pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    elif pooling == "avg":
        pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    else:
        raise ValueError("Unrecognized pooling argseter")
    for c in channel_list:
        if c == "P":
            layers += [pool2d]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


# fmt: off
channel_list = {
    "VGG-16p": [24, 22, "P", 41, 51, "P", 108, 89, 111, "P", 184, 276, 228, "P", 512, 512, 512, "P"],
    "VGG-16": [64, 64, "P", 128, 128, "P", 256, 256, 256, "P", 512, 512, 512, "P", 512, 512, 512, "P"],
    "VGG-19": [64, 64, "P", 128, 128, "P", 256, 256, 256, 256, "P", 512, 512, 512, 512, "P", 512, 512, 512, 512, "P"],
}
nin_dict = {
    "C": ["conv1", "cccp1", "cccp2", "conv2", "cccp3", "cccp4", "conv3", "cccp5", "cccp6", "conv4-1024", "cccp7-1024", "cccp8-1024"],
    "R": ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7", "relu8", "relu9", "relu10", "relu11", "relu12"],
    "P": ["pool1", "pool2", "pool3", "pool4"],
    "D": ["drop"],
}
vgg16_dict = {
    "C": ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"],
    "R": ["relu1_1", "relu1_2", "relu2_1", "relu2_2", "relu3_1", "relu3_2", "relu3_3", "relu4_1", "relu4_2", "relu4_3", "relu5_1", "relu5_2", "relu5_3"],
    "P": ["pool1", "pool2", "pool3", "pool4", "pool5"],
}
vgg19_dict = {
    "C": ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4", "conv5_1", "conv5_2", "conv5_3", "conv5_4"],
    "R": ["relu1_1", "relu1_2", "relu2_1", "relu2_2", "relu3_1", "relu3_2", "relu3_3", "relu3_4", "relu4_1", "relu4_2", "relu4_3", "relu4_4", "relu5_1", "relu5_2", "relu5_3", "relu5_4"],
    "P": ["pool1", "pool2", "pool3", "pool4", "pool5"],
}
vgg_list = ["fcn32s", "prun", "sod", "vgg", "nyud"]
# fmt: on


def select_model(model_file, pooling, verbose, disable_check):
    if any(name in model_file for name in vgg_list):
        if "prun" in model_file:
            if verbose:
                print("VGG-16 Architecture Detected")
                print("Using The Channel Pruning Model")
            if not path.exists(model_file):
                model_file = "modelzoo/vgg16-prune.pth"
                if not path.exists(model_file):
                    gdown.download("https://drive.google.com/uc?id=1aaNqJ5D2A-vev3IZFv6dSkovuA3XwYsq", model_file)
            cnn, layerList = VGG_PRUNED(build_sequential(channel_list["VGG-16p"], pooling)), vgg16_dict
        elif "nyud" in model_file:
            if verbose:
                print("VGG-16 Architecture Detected")
                print("Using the nyud-fcn32s-color-heavy Model")
            if not path.exists(model_file):
                model_file = "modelzoo/nyud-fcn32s-color-heavy.pth"
                if not path.exists(model_file):
                    gdown.download("https://drive.google.com/uc?id=1MKj6Dntzh7t45PxM4I0ixWaQtisAg9hy", model_file)
            cnn, layerList = VGG_FCN32S(build_sequential(channel_list["VGG-16"], pooling)), vgg16_dict
        elif "fcn32s" in model_file:
            if verbose:
                print("VGG-16 Architecture Detected")
                print("Using the fcn32s-heavy-pascal Model")
            if not path.exists(model_file):
                model_file = "modelzoo/fcn32s-heavy-pascal.pth"
                if not path.exists(model_file):
                    gdown.download("https://drive.google.com/uc?id=1bcAnvfMuuEbJqjaVWIUCD9HUgD1fvxI_", model_file)
            cnn, layerList = VGG_FCN32S(build_sequential(channel_list["VGG-16"], pooling)), vgg16_dict
        elif "sod" in model_file:
            if verbose:
                print("VGG-16 Architecture Detected")
                print("Using The SOD Finetune Model")
            if not path.exists(model_file):
                model_file = "modelzoo/vgg16-sod.pth"
                if not path.exists(model_file):
                    gdown.download("https://drive.google.com/uc?id=1EU-F9ugeIeTO9ay4PinzsBXgEuCYBu0Z", model_file)
            cnn, layerList = VGG_SOD(build_sequential(channel_list["VGG-16"], pooling)), vgg16_dict
        elif "vgg19" in model_file:
            if verbose:
                print("VGG-19 Architecture Detected")
            if not path.exists(model_file):
                # Download the VGG-19 model and fix the layer names
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
            cnn, layerList = VGG(build_sequential(channel_list["VGG-19"], pooling)), vgg19_dict
        elif "vgg16" in model_file:
            if verbose:
                print("VGG-16 Architecture Detected")
            if not path.exists(model_file):
                # Download the VGG-16 model and fix the layer names
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
            cnn, layerList = VGG(build_sequential(channel_list["VGG-16"], pooling)), vgg16_dict
        else:
            raise ValueError("VGG architecture not recognized.")
    elif "nin" in model_file:
        if verbose:
            print("NIN Architecture Detected")
        if not path.exists(model_file):
            # Download the NIN model
            model_file = "modelzoo/nin.pth"
            if not path.exists(model_file):
                print("Downloading...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", model_file
                )
        cnn, layerList = NIN(pooling), nin_dict
    else:
        raise ValueError("Model architecture not recognized.")

    cnn.load_state_dict(torch.load(model_file), strict=(not disable_check))
    if verbose:
        print("Successfully loaded " + str(model_file))

    return cnn, layerList


# Load the model, and configure pooling layer type
def load_model(args):
    cnn, layer_list = select_model(str(args.model_file).lower(), args.pooling, args.verbose, args.disable_check)

    # Maybe convert the model to cuda now, to avoid later issues
    if "c" not in str(args.gpu).lower() or "c" not in str(args.gpu[0]).lower():
        cnn = cnn.cuda()
    cnn = cnn.features

    content_layers = args.content_layers.split(",")
    style_layers = args.style_layers.split(",")

    # Set up the network, inserting style and content loss modules
    cnn = copy.deepcopy(cnn)
    content_losses, style_losses, tv_losses, temporal_losses = [], [], [], []
    next_content_idx, next_style_idx = 1, 1
    net = nn.Sequential()
    c, r = 0, 0

    if args.tv_weight > 0:
        tv_mod = TVLoss(args.tv_weight)
        tv_mod.name = f"tv {len(net)}"
        net.add_module(str(len(net)), tv_mod)
        tv_losses.append(tv_mod)

    if args.temporal_weight > 0:
        temporal_mod = ContentLoss(args.temporal_weight, args.normalize_gradients)
        temporal_mod.name = f"temporal {len(net)}"
        net.add_module(str(len(net)), temporal_mod)
        temporal_losses.append(temporal_mod)

    for i, layer in enumerate(list(cnn), 1):
        if next_content_idx <= len(content_layers) or next_style_idx <= len(style_layers):
            if isinstance(layer, nn.Conv2d):
                net.add_module(str(len(net)), layer)

                if layer_list["C"][c] in content_layers:
                    if args.verbose:
                        print("Setting up content layer " + str(i) + ": " + str(layer_list["C"][c]))
                    loss_module = ContentLoss(args.content_weight, args.normalize_gradients)
                    loss_module.name = f"cont {len(net)}"
                    net.add_module(str(len(net)), loss_module)
                    content_losses.append(loss_module)

                if layer_list["C"][c] in style_layers:
                    if args.verbose:
                        print("Setting up style layer " + str(i) + ": " + str(layer_list["C"][c]))
                    loss_module = StyleLoss(
                        args.style_weight,
                        args.use_covariance,
                        args.normalize_gradients,
                        video_style_factor=args.video_style_factor,
                        shift_factor=args.shift_factor,
                    )
                    loss_module.name = f"style {len(net)}"
                    net.add_module(str(len(net)), loss_module)
                    style_losses.append(loss_module)
                c += 1

            if isinstance(layer, nn.ReLU):
                net.add_module(str(len(net)), layer)

                if layer_list["R"][r] in content_layers:
                    if args.verbose:
                        print("Setting up content layer " + str(i) + ": " + str(layer_list["R"][r]))
                    loss_module = ContentLoss(args.content_weight, args.normalize_gradients)
                    loss_module.name = f"cont {len(net)}"
                    net.add_module(str(len(net)), loss_module)
                    content_losses.append(loss_module)
                    next_content_idx += 1

                if layer_list["R"][r] in style_layers:
                    if args.verbose:
                        print("Setting up style layer " + str(i) + ": " + str(layer_list["R"][r]))
                    loss_module = StyleLoss(
                        args.style_weight,
                        args.use_covariance,
                        args.normalize_gradients,
                        video_style_factor=args.video_style_factor,
                        shift_factor=args.shift_factor,
                    )
                    loss_module.name = f"style {len(net)}"
                    net.add_module(str(len(net)), loss_module)
                    style_losses.append(loss_module)
                    next_style_idx += 1
                r += 1

            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                net.add_module(str(len(net)), layer)

    if args.multidevice:
        net = setup_multi_device(net, args.gpu, args.multidevice_strategy)

    # Freeze the network in order to prevent unnecessary gradient calculations
    for param in net.parameters():
        param.requires_grad = False

    # monkey patch our losses onto the network for easy access
    net.content_losses = content_losses
    net.style_losses = style_losses
    net.tv_losses = tv_losses
    net.temporal_losses = temporal_losses

    return net, content_losses + style_losses + tv_losses + temporal_losses


class NewModelParallel(nn.Module):
    def __init__(self, net, device_ids, device_splits):
        super(NewModelParallel, self).__init__()
        self.device_list = self.name_devices(device_ids.split(","))
        self.chunks = self.chunks_to_devices(self.split_net(net, device_splits.split(",")))

    def name_devices(self, input_list):
        device_list = []
        for i, device in enumerate(input_list):
            if str(device).lower() != "c":
                device_list.append("cuda:" + str(device))
            else:
                device_list.append("cpu")
        return device_list

    def split_net(self, net, device_splits):
        chunks, cur_chunk = [], nn.Sequential()
        for i, l in enumerate(net):
            cur_chunk.add_module(str(i), net[i])
            if str(i) in device_splits and device_splits != "":
                del device_splits[0]
                chunks.append(cur_chunk)
                cur_chunk = nn.Sequential()
        chunks.append(cur_chunk)
        return chunks

    def chunks_to_devices(self, chunks):
        for i, chunk in enumerate(chunks):
            chunk.to(self.device_list[i])
        return chunks

    def c(self, input, i):
        if input.type() == "torch.FloatTensor" and "cuda" in self.device_list[i]:
            input = input.type("torch.cuda.FloatTensor")
        elif input.type() == "torch.cuda.FloatTensor" and "cpu" in self.device_list[i]:
            input = input.type("torch.FloatTensor")
        return input

    def forward(self, input):
        for i, chunk in enumerate(self.chunks):
            if i < len(self.chunks) - 1:
                input = self.c(chunk(self.c(input, i).to(self.device_list[i])), i + 1).to(self.device_list[i + 1])
            else:
                input = chunk(input)
        return input


class ModelParallel(nn.Module):
    def __init__(self, chunks, device_list):
        super(ModelParallel, self).__init__()
        self.chunks = chunks
        self.device_list = device_list

    def __str__(self):
        return "".join([f"device {d}:\n{c.__str__}\n" for d, c in zip(self.device_list, self.chunks)])

    def c(self, input, i):
        if input.type() == "torch.FloatTensor" and "cuda" in self.device_list[i]:
            input = input.type("torch.cuda.FloatTensor")
        elif input.type() == "torch.cuda.FloatTensor" and "cpu" in self.device_list[i]:
            input = input.type("torch.FloatTensor")
        return input

    def forward(self, input):
        for i, chunk in enumerate(self.chunks):
            if i < len(self.chunks) - 1:
                input = self.c(chunk(self.c(input, i).to(self.device_list[i])), i + 1).to(self.device_list[i + 1])
            else:
                input = chunk(input)
        return input


def setup_new_multi_device(net, gpu, multidevice_strategy):
    assert len(str(gpu).split(",")) - 1 == len(
        str(multidevice_strategy).split(",")
    ), "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."

    new_net = NewModelParallel(net, str(gpu), str(multidevice_strategy))
    return new_net


def setup_multi_device(net, gpu, multidevice_strategy):
    device_splits = str(multidevice_strategy).split(",")
    gpu = str(gpu).split(",")

    assert len(gpu) - 1 == len(
        device_splits
    ), "The number of -multidevice_strategy layer indices must be equal to the number of -gpu devices minus 1."

    device_list = []
    for i, device in enumerate(gpu):
        if str(device).lower() != "c":
            device_list.append("cuda:" + str(device))
        else:
            device_list.append("cpu")

    cur_chunk = nn.Sequential()
    chunks = []
    for i, l in enumerate(net):
        cur_chunk.add_module(str(i), net[i])
        if str(i) in device_splits and device_splits != "":
            del device_splits[0]
            chunks.append(cur_chunk)
            cur_chunk = nn.Sequential()
    chunks.append(cur_chunk)

    for i, chunk in enumerate(chunks):
        chunk.to(device_list[i])

    new_net = ModelParallel(chunks, device_list)
    return new_net
