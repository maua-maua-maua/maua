from torch.optim import LBFGS


def load_optimizer(name):
    if name == "lbfgs":
        return LBFGS
