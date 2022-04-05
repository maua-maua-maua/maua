import torch


class DiscriminatorCrossEntropy(torch.nn.Module):
    def __init__(self, logits=True) -> None:
        super().__init__()
        self.cross_entropy = torch.nn.BCEWithLogitsLoss() if logits else torch.nn.BCELoss()

    def forward(self, lightning_module, real_preds, fake_preds, **kwargs):
        loss_D_real = self.cross_entropy(real_preds, torch.ones_like(real_preds))
        loss_D_fake = self.cross_entropy(fake_preds, torch.zeros_like(fake_preds))

        lightning_module.log_dict(
            dict(
                loss_D_real=loss_D_real,
                loss_D_fake=loss_D_fake,
                real_preds=real_preds.mean(),
                fake_preds=fake_preds.mean(),
            )
        )

        return loss_D_real + loss_D_fake


class GeneratorCrossEntropy(torch.nn.Module):
    def __init__(self, logits=True) -> None:
        super().__init__()
        self.cross_entropy = torch.nn.BCEWithLogitsLoss() if logits else torch.nn.BCELoss()

    def forward(self, lightning_module, fake_preds, **kwargs):
        loss_G = self.cross_entropy(fake_preds, torch.ones_like(fake_preds))

        lightning_module.log_dict(dict(loss_G=loss_G))

        return loss_G
