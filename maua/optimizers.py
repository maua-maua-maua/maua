from copy import deepcopy

import timm.optim as timm_optim
import torch_optimizer as more_optim
import pytorch_optimizer as even_more_optim
from torch import optim

optimizer_choices = {
    "AccSGD": more_optim.AccSGD,
    "AdaBelief": timm_optim.AdaBelief,
    "AdaBound": more_optim.AdaBound,
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "Adahessian": timm_optim.Adahessian,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "Adamax": optim.Adamax,
    "AdaMod": more_optim.AdaMod,
    "Adafactor": more_optim.Adafactor,
    "AdamP": more_optim.AdamP,
    "AggMo": more_optim.AggMo,
    "DiffGrad": more_optim.DiffGrad,
    "FusedSGD": timm_optim.optim_factory.FusedSGD,
    "FusedAdam": timm_optim.optim_factory.FusedAdam,
    "FusedLAMB": timm_optim.optim_factory.FusedLAMB,
    "FusedNovoGrad": timm_optim.optim_factory.FusedNovoGrad,
    "Lamb": more_optim.Lamb,
    "NAdam": optim.NAdam,
    "NovoGrad": more_optim.NovoGrad,
    "NvNovoGrad": timm_optim.NvNovoGrad,
    "PID": more_optim.PID,
    "QHAdam": more_optim.QHAdam,
    "QHM": more_optim.QHM,
    "RAdam": more_optim.RAdam,
    "Ranger": more_optim.Ranger,
    "RangerQH": more_optim.RangerQH,
    "RangerVA": more_optim.RangerVA,
    "Ranger21": even_more_optim.Ranger21,
    "RMSprop": optim.RMSprop,
    "RMSpropTF": timm_optim.RMSpropTF,
    "SGD": optim.SGD,
    "SGDP": more_optim.SGDP,
    "SGDW": more_optim.SGDW,
    "Shampoo": more_optim.Shampoo,
    "SWATS": more_optim.SWATS,
    "Yogi": more_optim.Yogi,
}
OPTIMIZERS = list(optimizer_choices.keys()) + ["LBFGS"]


def load_optimizer(name, lr, kwargs, n_iters, params):
    if name == "LBFGS":
        if kwargs == {}:
            kwargs = dict(tolerance_grad=-1, tolerance_change=-1)
        return optim.LBFGS(params, lr=lr, max_iter=n_iters, **kwargs), 1

    if "LBFGS-" in name:
        max_iter = int(name.split("-")[1])
        return optim.LBFGS(params, lr=lr, max_iter=max_iter, **kwargs), n_iters // max_iter

    if name in optimizer_choices:
        return optimizer_choices[name](params, lr=lr, **kwargs), n_iters

    if "Lookahead-" in name:
        if "lookahead_alpha" in kwargs:
            alpha = deepcopy(kwargs["lookahead_alpha"])
            del kwargs["lookahead_alpha"]
        else:
            alpha = 0.5

        if "lookahead_k" in kwargs:
            k = deepcopy(kwargs["lookahead_k"])
            del kwargs["lookahead_k"]
        else:
            k = 0.5

        return timm_optim.Lookahead(load_optimizer(name.split("-")[-1], lr, kwargs, n_iters, params), alpha, k)

    raise Exception(
        f"Optimizer {name} not recognized! Choices are: {['LBFGS', 'LBFGS20'] + list(optimizer_choices.keys())}"
    )
