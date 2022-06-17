import torch

from . import Loss


class DiscriminatorCrossEntropy(Loss):
    def __init__(self, logits=True, **kwargs) -> None:
        super().__init__()
        self.cross_entropy = (
            torch.nn.BCEWithLogitsLoss(reduction="none") if logits else torch.nn.BCELoss(reduction="none")
        )

    def forward(self, lightning_module, preds_real, preds_fake, **kwargs):
        loss_D_real = self.cross_entropy(preds_real, torch.ones_like(preds_real))
        loss_D_fake = self.cross_entropy(preds_fake, torch.zeros_like(preds_fake))

        lightning_module.log_dict(
            dict(
                loss_D_real=loss_D_real.mean(),
                loss_D_fake=loss_D_fake.mean(),
                preds_real=preds_real.sign().mean(),
                preds_fake=preds_fake.sign().mean(),
            )
        )

        return loss_D_real + loss_D_fake


class GeneratorCrossEntropy(Loss):
    def __init__(self, logits=True, **kwargs) -> None:
        super().__init__()
        self.cross_entropy = (
            torch.nn.BCEWithLogitsLoss(reduction="none") if logits else torch.nn.BCELoss(reduction="none")
        )

    def forward(self, lightning_module, preds_fake, **kwargs):
        loss_G = self.cross_entropy(preds_fake, torch.ones_like(preds_fake))

        lightning_module.log_dict(dict(loss_G=loss_G.mean()))

        return loss_G
