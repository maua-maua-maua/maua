import numpy as np
import torch

from .jacnorm import jacobian_norm_rejection
from .langevin import langevin_with_critic
from .polarity import polarity_sampling


def sample_latents(G, seeds, batch_size, truncation, how, langevin_critic):
    base_zs = torch.cat([torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)) for seed in seeds]).float()

    if how.startswith("langevin"):
        return langevin_with_critic(G, base_zs, langevin_critic, bs=batch_size)

    elif how == "jacobian":
        return jacobian_norm_rejection(G, base_zs, ratio=truncation)

    elif how == "polarity":
        raise NotImplementedError()
        G_path = Path(G.model_file)
        return polarity_sampling(  # TODO fix API to work with zs
            G, base_zs, polarity=truncation, cache_file=G_path.parent / f"{G_path.stem}_polarity_svds.npz"
        )

    else:
        return base_zs
