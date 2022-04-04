from typing import Generator

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm

torch._C._set_cublas_allow_tf32(True)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.jit.optimized_execution(True)
torch.jit.fuser("fuser2")


class MauaMapper(torch.nn.Module):
    def forward(self):
        raise NotImplementedError()


class MauaSynthesizer(torch.nn.Module):
    _hook_handles = []

    def forward(self):
        raise NotImplementedError()

    def change_output_resolution(self):
        raise NotImplementedError()

    def refresh_model_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


class MauaGenerator(torch.nn.Module):
    MapperCls = None
    SynthesizerCls = None

    def __init__(self, mapper_kwargs={}, synthesizer_kwargs={}) -> None:
        super().__init__()
        self.mapper = self.__class__.MapperCls(**mapper_kwargs)
        self.synthesizer = self.__class__.SynthesizerCls(**synthesizer_kwargs)

    def forward(self):
        raise NotImplementedError()

    def render(
        self,
        inputs,
        batch_size=32,
        postprocess_fn=lambda x: x,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        fp16=True,
        batched=True,
        verbose=False,
    ) -> Generator[torch.Tensor, None, None]:

        dataset = TensorDataset(*(i.cpu().pin_memory(device) for i in inputs.values()))

        def collate_fn(batch):
            return dict(
                zip(
                    inputs.keys(),
                    (b.to(device, dtype=torch.float16 if fp16 else torch.float32) for b in default_collate(batch)),
                )
            )

        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        self.to(device)

        if fp16:

            def force_half(mod):
                if hasattr(mod, "use_fp16"):
                    mod.use_fp16 = True
                if hasattr(mod, "noise_const"):
                    setattr(mod, "noise_const", mod.noise_const.half())

            self.mapper = self.mapper.half()
            # self.synthesizer = self.synthesizer.half()
            self.synthesizer = self.synthesizer.apply(force_half)

        if verbose:
            pbar = tqdm(loader, smoothing=0.8, unit_scale=batch_size, unit="img")
        else:
            pbar = loader
        for batch in pbar:
            frame_batch = self.synthesizer.forward(**batch).add(1).div(2).clamp(0, 1)
            frame_batch = postprocess_fn(frame_batch)
            if batched:
                yield frame_batch
            else:
                for frame in frame_batch:
                    yield frame[None]


def get_generator_class(architecture: str) -> MauaGenerator:
    if architecture == "stylegan3":
        from .stylegan3 import StyleGAN3

        return StyleGAN3
    if architecture == "stylegan2":
        from .stylegan2 import StyleGAN2

        return StyleGAN2
    else:
        raise Exception(f"Architecture not found: {architecture}")
