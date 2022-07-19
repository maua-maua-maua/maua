import torch

from . import Loss


class DiscriminatorSoftPlus(Loss):
    def forward(self, lightning_module, preds_real, preds_fake, **kwargs):
        loss_D_real = torch.nn.functional.softplus(-preds_real)
        loss_D_fake = torch.nn.functional.softplus(preds_fake)

        lightning_module.log_dict(
            dict(
                loss_D_real=loss_D_real.mean(),
                loss_D_fake=loss_D_fake.mean(),
                preds_real=preds_real.sign().mean(),
                preds_fake=preds_fake.sign().mean(),
            )
        )

        return loss_D_real + loss_D_fake


class GeneratorSoftPlus(Loss):
    def forward(self, lightning_module, preds_fake, **kwargs):
        loss_G = torch.nn.functional.softplus(-preds_fake)

        lightning_module.log_dict(dict(loss_G=loss_G.mean()))

        return loss_G
