import os
import random
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from maua.GAN.load import load_network

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/nv/")
from .nv.networks.stylegan2 import MappingNetwork, SynthesisBlock, SynthesisNetwork
from .wrappers import get_generator_class
from .wrappers.stylegan2 import StyleGAN2, StyleGAN2Mapper, StyleGAN2Synthesizer


def get_state_dict_key_levels(generator):
    def name_modules(module):
        def name_module(module, prefix=""):
            module.name = prefix[:-1]
            for name, child in module._modules.items():
                if child is not None:
                    name_module(child, prefix + name + ".")

        name_module(module)

    name_modules(generator)

    blend_modules, hooks = [], []

    def register_hook(module):
        def hook(module, input, output):
            if len(list(module.parameters())) > 0:
                blend_modules.append((module.name, module))

        if (
            not isinstance(module, torch.nn.Sequential)
            and not isinstance(module, torch.nn.ModuleList)
            and not isinstance(module, StyleGAN2Mapper)
            and not isinstance(module, MappingNetwork)
            and not isinstance(module, SynthesisBlock)
            and not isinstance(module, StyleGAN2Synthesizer)
            and not isinstance(module, SynthesisNetwork)
            and not isinstance(module, StyleGAN2)
        ):
            hooks.append(module.register_forward_hook(hook))

    generator.apply(register_hook)
    generator(torch.randn(1, 512, device="cuda"))
    [hook.remove() for hook in hooks]

    levels = []
    for mod_name, _ in blend_modules:
        if "map" in mod_name:
            level = 0
        else:
            key = mod_name.replace("synthesizer.G_synth.b", "").split(".")
            size, conv = key[0], key[1]
            level = int(np.log2(int(size))) - 2
            level *= 2
            level += conv != "conv0"
        levels.append(level)

    key_levels = {}
    for (mod_name, mod), level in zip(blend_modules, levels):
        for param_name, _ in mod.named_parameters():
            k = mod_name + "." + param_name
            key_levels[k] = level

    return key_levels


def get_blend_weights(midpoints, width, n_layers):
    level_idxs = torch.arange(n_layers, device=midpoints.device)
    relative_idxs = level_idxs[None, :] - midpoints[:, None]
    if width:
        blend_weights = 1 / (1 + torch.exp(-relative_idxs / width))
    else:
        blend_weights = relative_idxs > 1
    return blend_weights.float()


if __name__ == "__main__":
    checkpoint_dirs = [
        # "/home/hans/modelzoo/koanGAN/tjalack",
        "/home/hans/modelzoo/koanGAN/quoxal",
        "/home/hans/modelzoo/koanGAN/scifi",
        "/home/hans/modelzoo/koanGAN/sylvaleonsce",
        "/home/hans/modelzoo/koanGAN/koancept-v1",
    ]
    sample_strategy = "uniform"  # "bimodal", 'uniform'
    blend_strategy = "random"  # 'crossover'
    number = 10
    num_ckpts = 10
    out_dir = "/home/hans/modelzoo/koanGAN/blend/"
    architecture = "stylegan2"

    all_checkpoints = [glob(f"{ckptdir}/*.pt") + glob(f"{ckptdir}/*.pkl") for ckptdir in checkpoint_dirs]

    with torch.inference_mode():
        generator = get_generator_class(architecture)(model_file=None, output_size=(1024, 1024)).cuda()
        levels = get_state_dict_key_levels(generator)

        for _ in range(number):
            if sample_strategy == "random":
                checkpoints = random.choices(sum(all_checkpoints, []), k=num_ckpts)
            if sample_strategy == "uniform":
                checkpoints = [random.choice(random.choice(all_checkpoints)) for _ in range(num_ckpts)]
            elif sample_strategy == "ab":
                raise NotImplementedError()
                nets = {Path(ckpt).stem.split("-")[0] for ckpt in all_checkpoints}
                for a in ["diffuse"] * 5:
                    for b in nets - {a} - {"lakspe"}:

                        num_a = random.randint(4, 8)
                        num_b = random.randint(2, 3)

                        checkpoints = [
                            *random.choices(list(filter(lambda x: a in x, all_checkpoints)), k=num_a),
                            *random.choices(list(filter(lambda x: b in x, all_checkpoints)), k=num_b),
                        ]

            name = "_".join(
                [
                    re.sub("-batch[0-9]+", "", re.sub("-gpus[0-9]+", "", Path(p).stem))
                    .replace("-1024", "")
                    .replace("-stylegan2", "")
                    .replace("network-snapshot-", "")
                    for p in checkpoints
                ]
            )[:222]
            print(name)

            if blend_strategy == "crossover":
                mix_types = torch.randint(0, 3, (len(checkpoints),))
                weights = [
                    get_blend_weights(
                        midpoints=torch.randint(-1, generator.n_latent + 1, (1,)),
                        width=torch.rand(1) * generator.n_latent / 2,
                        n_layers=generator.n_latent,
                    ).squeeze()
                    for _ in range(len(checkpoints))
                ]

            # perform blending
            state_dict, state_weight = {}, {}
            for c, checkpoint_path in enumerate(checkpoints):
                checkpoint = get_generator_class(architecture)(model_file=checkpoint_path).state_dict()

                for key, val in checkpoint.items():
                    if not key in state_dict:
                        state_dict[key] = torch.zeros_like(val)
                        state_weight[key] = 0

                    if blend_strategy == "random":
                        weight = torch.rand([])
                        state_dict[key] += weight * val
                        state_weight[key] += weight

                    if blend_strategy == "crossover":
                        mix = mix_types[c]
                        weight = weights[c]
                        if mix == 0:
                            state_dict[key] += val
                            state_weight[key] += 1
                        elif mix == 1:
                            level = levels.get(key, "not found")
                            if level == "not found":
                                strength = 1
                            else:
                                strength = weight[level]
                            state_dict[key] += strength * val
                            state_weight[key] += strength
                        elif mix == 2:
                            level = levels.get(key, "not found")
                            if level == "not found":
                                strength = 0
                            else:
                                strength = 1 - weight[level]
                            state_dict[key] += strength * val
                            state_weight[key] += strength
            for k, w in state_weight.items():
                state_dict[k] /= w

            # evaluate new checkpoint
            generator.load_state_dict(state_dict)
            generator.eval()
            sample_z = torch.randn((60, 512)).cuda()
            sample = []
            for sub in range(0, len(sample_z), 8):
                subsample = generator(sample_z[sub : sub + 8])
                sample.append(subsample.detach().cpu())
            sample = torch.cat(sample).detach()

            torchvision.utils.save_image(
                sample,
                f"{out_dir}/{name}-{sample_strategy}sample-{blend_strategy}blend.jpg",
                nrow=10,
                normalize=True,
                value_range=(-1, 1),
            )
            torch.save(
                {
                    "G_ema": {
                        k.replace("mapper.G_map", "mapping").replace("synthesizer.G_synth", "synthesis"): v
                        for k, v in generator.state_dict().items()
                    }
                },
                f"{out_dir}/{name}-{sample_strategy}sample-{blend_strategy}blend.pt",
            )
