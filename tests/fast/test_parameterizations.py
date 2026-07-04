import pytest
import torch


def test_load_parameterization_unknown():
    from maua.parameterizations import load_parameterization

    with pytest.raises(Exception):
        load_parameterization("does-not-exist")


def test_rgb_decode_shape():
    from maua.parameterizations import load_parameterization

    RGB = load_parameterization("rgb")
    param = RGB(16, 16, tensor=torch.rand(1, 3, 16, 16))
    img = param.decode()
    assert img.shape[-2:] == (16, 16) and img.shape[1] == 3
