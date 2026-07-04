import torch

from maua.prompt import ContentPrompt, ImagePrompt, StylePrompt, TextPrompt


def test_text_prompt():
    text, weight = TextPrompt("a test prompt", weight=2.0)()
    assert text == "a test prompt" and weight == 2.0


def test_image_prompt(example_image):
    img, weight = ImagePrompt(path=str(example_image))()
    assert torch.is_tensor(img) and img.ndim == 4
    assert -1 <= img.min() and img.max() <= 1


def test_image_prompt_resize(example_image):
    img, _ = ImagePrompt(path=str(example_image), size=(32, 32))()
    assert img.shape[-2:] == (32, 32)


def test_style_content_prompts(example_image):
    assert isinstance(StylePrompt(path=str(example_image)), ImagePrompt)
    assert isinstance(ContentPrompt(path=str(example_image)), ImagePrompt)
