import os

import torch
from maua.utility import download


def load_aesthetic_model():
    if not os.path.exists("modelzoo/ava_vit_b_16_full.pth"):
        download(
            "https://v-diffusion.s3.us-west-2.amazonaws.com/ava_vit_b_16_full.pth", "modelzoo/ava_vit_b_16_full.pth"
        )
    aesthetic_model = torch.nn.Linear(512, 10)
    aesthetic_model.load_state_dict(torch.load("modelzoo/ava_vit_b_16_full.pth"))
    return aesthetic_model


global aesthetic_model
aesthetic_model = None


def aesthetic_score(clip_embedding, target=8, scale=8, expected=False):
    global aesthetic_model
    if aesthetic_model is None:
        aesthetic_model = load_aesthetic_model()

    if expected:
        probs = torch.nn.functional.softmax(aesthetic_model(clip_embedding))
        expected = (probs * (1 + torch.arange(10, device=clip_embedding.device))).sum(-1)
        return -(scale * expected.mean(0)).sum()

    else:
        log_probs = torch.nn.functional.log_softmax(aesthetic_model(clip_embedding))
        return -(scale * log_probs[:, :, target - 1].mean(0)).sum()
