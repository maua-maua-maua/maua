import torch

from maua.ops.image import resample

from ... import audioreactive as ar


class MauaPatch:
    """ """

    def __init__(self, audio_file, fps=24, offset=0, duration=-1) -> None:
        self.fps = fps
        self.audio, self.sr, self.duration = ar.load_audio(audio_file, offset, duration)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_frames = round(self.duration * self.fps)

    def process_audio(self):
        pass

    def force_output_size(self, video):
        t, c, h, w = video.shape
        if (w, h) != self.synthesizer.output_size:
            video = resample(video, reversed(self.synthesizer.output_size))
        return video


def get_patch_from_file(filepath):
    import importlib
    import inspect

    module_name = filepath.replace(".py", "").replace("/", ".")

    for _, cls in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
        if cls.__module__ == module_name and issubclass(cls, MauaPatch):
            return cls

    raise Exception("Patch not found! Are you sure there is a class that extends MauaPatch in the file you specified?")
