import torch_optimizer as more_optim
from torch import optim

optimizer_choices = {
    "AccSGD": more_optim.AccSGD,
    "AdaBound": more_optim.AdaBound,
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "Adamax": optim.Adamax,
    "AdaMod": more_optim.AdaMod,
    "Adafactor": more_optim.Adafactor,
    "AdamP": more_optim.AdamP,
    "AggMo": more_optim.AggMo,
    "DiffGrad": more_optim.DiffGrad,
    "Lamb": more_optim.Lamb,
    "NAdam": optim.NAdam,
    "NovoGrad": more_optim.NovoGrad,
    "PID": more_optim.PID,
    "QHAdam": more_optim.QHAdam,
    "QHM": more_optim.QHM,
    "RAdam": more_optim.RAdam,
    "RMSprop": optim.RMSprop,
    "SGD": optim.SGD,
    "SGDP": more_optim.SGDP,
    "SGDW": more_optim.SGDW,
    "Shampoo": more_optim.Shampoo,
    "SWATS": more_optim.SWATS,
    "Yogi": more_optim.Yogi,
}


def load_optimizer(name, kwargs, num_iters, params):
    if name == "LBFGS":
        return optim.LBFGS(params, max_iter=num_iters, **kwargs), 1

    if name == "LBFGS-20":
        return optim.LBFGS(params, max_iter=20, **kwargs), num_iters // 20

    if name in optimizer_choices:
        return optimizer_choices[name](params, **kwargs), num_iters

    raise Exception(
        f"Optimizer {name} not recognized! Choices are: {['LBFGS', 'LBFGS20'] + list(optimizer_choices.keys())}"
    )
