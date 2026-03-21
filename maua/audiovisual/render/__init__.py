import torch


class Renderer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_output_class(renderer):
    if renderer == "memmap":
        from maua.audiovisual.render.memmap import MemMap

        return MemMap
    if renderer == "ffmpeg":
        from maua.audiovisual.render.ffmpeg import FFMPEG

        return FFMPEG
    raise NotImplementedError
