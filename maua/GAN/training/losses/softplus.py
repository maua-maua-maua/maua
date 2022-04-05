import torch


class DiscriminatorSoftPlus(torch.nn.Module):
    def forward(self, lightning_module, real_preds, fake_preds, **kwargs):
        loss_D_real = torch.nn.functional.softplus(-real_preds).sum()
        loss_D_fake = torch.nn.functional.softplus(fake_preds).sum()

        lightning_module.log_dict(
            dict(
                loss_D_real=loss_D_real,
                loss_D_fake=loss_D_fake,
                real_preds=real_preds.mean(),
                fake_preds=fake_preds.mean(),
            )
        )

        return loss_D_real + loss_D_fake


class GeneratorSoftPlus(torch.nn.Module):
    def forward(self, lightning_module, fake_preds, **kwargs):
        loss_G = torch.nn.functional.softplus(-fake_preds).sum()

        lightning_module.log_dict(dict(loss_G=loss_G))

        return loss_G
