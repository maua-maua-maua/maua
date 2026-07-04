import pytest

pytestmark = [pytest.mark.gpu]


@pytest.mark.xfail(reason="nca/train.py train() reads module-level globals (style_file); repair in Phase B")
def test_nca_train():
    from maua.nca.train import train

    train()


@pytest.mark.xfail(reason="nca/generate.py imports nonexistent NCA_train and undefined globals; repair in Phase B")
def test_nca_generate():
    import maua.nca.generate  # noqa: F401
