import os

import numpy as np
from npy_append_array import NpyAppendArray
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import Renderer


class MemMap(Renderer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, synthesizer, inputs, postprocess):
        dataset = TensorDataset(*inputs.values())

        def collate_fn(data):
            return {k: v.unsqueeze(0).to(self.device) for k, v in zip(inputs.keys(), data[0])}

        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        synthesizer = synthesizer.to(self.device)

        cache_file = "workspace/frames_memmap.npy"
        if os.path.exists(cache_file):
            os.remove(cache_file)
        with NpyAppendArray(cache_file) as frames:

            for inputs in tqdm(loader):
                frame = synthesizer(**inputs)
                frames.append(frame.add(1).div(2).clamp(0, 1).mul(255).cpu().numpy().astype(np.uint8))

        frames = np.load(cache_file, mmap_mode="r")
        return postprocess(frames)
