import torch
from decord import VideoReader
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from ...ops.video import VideoWriter
from . import Renderer

torch._C._set_cublas_allow_tf32(True)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.jit.optimized_execution(True)
torch.jit.fuser("fuser2")


class FFMPEG(Renderer):
    def __init__(
        self,
        output_file,
        fps=24,
        audio_file=None,
        audio_offset=0,
        audio_duration=None,
        ffmpeg_preset="medium",
        batch_size=16,
    ):
        super().__init__()
        self.output_file, self.fps, self.ffmpeg_preset = output_file, fps, ffmpeg_preset
        self.audio_file, self.audio_offset, self.audio_duration = audio_file, audio_offset, audio_duration
        self.batch_size = batch_size

    def __call__(self, synthesizer, inputs, postprocess, fp16=True):

        dataset = TensorDataset(*(i.cpu().pin_memory(self.device) for i in inputs.values()))

        def collate_fn(batch):
            return dict(
                zip(
                    inputs.keys(),
                    (b.to(self.device, dtype=torch.float16 if fp16 else torch.float32) for b in default_collate(batch)),
                )
            )

        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        synthesizer = synthesizer.to(self.device)

        if fp16:

            def force_half(mod):
                if hasattr(mod, "use_fp16"):
                    mod.use_fp16 = True
                if hasattr(mod, "noise_const"):
                    setattr(mod, "noise_const", mod.noise_const.half())

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
            for batch in tqdm(loader, desc="Rendering to file...", unit_scale=self.batch_size, unit="img"):
                frame_batch = synthesizer(**batch).add(1).div(2)
                frame_batch = postprocess(frame_batch)
                for frame in frame_batch:
                    video.write(frame[None])

        # return VideoReader(self.output_file)
