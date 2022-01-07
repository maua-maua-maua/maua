import ffmpeg
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import Renderer


class FFMPEG(Renderer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, synthesizer, inputs):
        dataset = TensorDataset(*inputs.values())

        def collate_fn(data):
            return {k: v.to(self.device) for k, v in zip(inputs.keys(), data[0])}

        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        frames = []
        for inputs in tqdm(loader):
            frame = synthesizer(**inputs)
            frames.append(frame.cpu())

        return torch.cat(frames)
