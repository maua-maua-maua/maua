import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def langevin_sampling(
    zs,
    G,
    D,
    rate=0.01,  # an initial update rate for langevin sampling
    noise_std=1,  # standard deviation of a gaussian noise used in langevin sampling
    decay=0.1,  # decay strength for rate and noise_std
    decay_steps=200,  # rate and noise_std decrease every decay_steps
    steps=1000,  # total steps of langevin sampling
):
    scaler = 1.0
    apply_decay = decay > 0 and decay_steps > 0
    mean = torch.zeros(zs.shape[1], device=zs.device)
    prior_std = torch.eye(zs.shape[1], device=zs.device)
    lgv_std = prior_std * noise_std
    prior = MultivariateNormal(loc=mean, covariance_matrix=prior_std)
    lgv_prior = MultivariateNormal(loc=mean, covariance_matrix=lgv_std)

    for i in range(steps):
        zs = zs.requires_grad_()
        fake_images = G(zs)
        fake_logits = D(fake_images)

        energy = -prior.log_prob(zs) - fake_logits
        z_grads = torch.autograd.grad(
            outputs=energy,
            inputs=zs,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        zs = zs - 0.5 * rate * z_grads + (rate**0.5) * lgv_prior.sample([zs.shape[0]]) * scaler

        if apply_decay and (i + 1) % decay_steps == 0:
            rate *= decay
            scaler *= decay

    return zs
