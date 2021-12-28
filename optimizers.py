from torch.optim import LBFGS


def load_optimizer(name, kwargs, num_iters, params):
    if name == "lbfgs":
        return LBFGS(params, max_iter=num_iters, **kwargs), 1
    if name == "lbfgs20":
        return LBFGS(params, **kwargs), num_iters // 20
