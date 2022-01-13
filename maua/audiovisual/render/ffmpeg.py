import torch
from decord import VideoReader
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from maua.ops.tensor import tensor2bytes
from maua.ops.video import VideoWriter

from . import Renderer


class FFMPEG(Renderer):
    def __init__(self, output_file, fps=24, audio_file=None, audio_offset=0, audio_duration=None, ffmpeg_preset="slow"):
        super().__init__()
        self.output_file, self.fps, self.ffmpeg_preset = output_file, fps, ffmpeg_preset
        self.audio_file, self.audio_offset, self.audio_duration = audio_file, audio_offset, audio_duration

    def __call__(self, synthesizer, inputs, postprocess, fp16=False):
        dataset = TensorDataset(*inputs.values())

        def collate_fn(data):
            return {k: v.unsqueeze(0).to(self.device) for k, v in zip(inputs.keys(), data[0])}

        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        synthesizer = synthesizer.to(self.device)

        if fp16:

            def force_half(mod):
                if hasattr(mod, "use_fp16"):
                    mod.use_fp16 = True

            synthesizer.G_synth.apply(force_half)

        with VideoWriter(
            self.output_file,
            synthesizer.output_size,
            self.fps,
            self.audio_file,
            self.audio_offset,
            self.audio_duration,
            self.ffmpeg_preset,
        ) as video:
            for batch in tqdm(loader):
                frame = synthesizer(**batch).add(1).div(2)
                frame = postprocess(frame)
                video.write(tensor2bytes(frame.squeeze()).tobytes())

        return VideoReader(self.output_file)
