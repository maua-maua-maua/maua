from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, cast

from bidict import bidict
from tensordict import TensorDict
from torch.nn import Module, ModuleDict, ModuleList

from maua2.processor import Processor

R = TypeVar("R")


class Attribute:
    pass


def abstractattribute(obj: Optional[Callable[[Any], R]] = None) -> R:
    _obj = cast(Any, obj)
    if obj is None:
        _obj = Attribute()
    _obj.__is_abstract_attribute__ = True  # type: ignore
    return cast(R, _obj)


class ProcessorMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = ABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name for name in dir(instance) if getattr(getattr(instance, name), "__is_abstract_attribute__", False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                f"{type(instance).__name__} must implement the `in_dims` and `out_dims` attributes! These attributes "
                "must be dictionaries mapping input/output names to their number of dimensions (excluding the batch "
                "dimension).\n\nFor example:\n  - a processor that takes an RGB image input named 'x' and outputs a "
                "video named 'y' would have `in_dims = {'x': 3}` (channels, height, width) and `out_dims = {'y': 4}` "
                "(time, channels, height, width).\n  - a processor that takes a text prompt named 'a' and a sine wave "
                "named 'b' and outputs a sequence of latent vectors named 'c' would have `in_dims = {'a': 1, 'b': 1}` "
                "(sequence length) and `out_dims = {'c': 2}` (sequence length, latent dimension).\n\n"
            )
        return instance


class Processor(Module, metaclass=ProcessorMeta):
    def __init__(self) -> None:
        super().__init__()
        self.incoming: Set[Tuple[str, Processor]] = set()
        self.outgoing: Set[Tuple[str, Processor]] = set()

    @abstractattribute
    def in_dims(self) -> Dict[str, Tuple[int, ...]]: ...

    @abstractattribute
    def out_dims(self) -> Dict[str, Tuple[int, ...]]: ...

    def inputs(self) -> Tuple[str, ...]:
        return tuple(self.in_dims.keys())

    def outputs(self) -> Tuple[str, ...]:
        return tuple(self.out_dims.keys())

    def connect(self, target: "Processor", source_field: str, target_field: str):
        self.outgoing.add((source_field, target))
        target.incoming.add((target_field, self))

    @abstractmethod
    def forward(self, inputs: TensorDict) -> TensorDict: ...


class Graph(Module):
    def __init__(
        self,
        processors: Dict[str, Processor],
        connections: Dict[Tuple[str, str], Tuple[str, str]],
    ):
        super().__init__()

        self.processors: Dict[str, Processor] = ModuleDict(processors)  # type: ignore

        self.connections: bidict[Tuple[str, str], Tuple[str, str]] = bidict()

        for (source_name, source_field), (
            target_name,
            target_field,
        ) in connections.items():
            self.connect(source_name, source_field, target_name, target_field)

    def connect(self, source_name: str, source_field: str, target_name: str, target_field: str):
        if source_name not in self.processors:
            raise ValueError(f"ProcessorGraph has no processor named '{source_name}'")
        if target_name not in self.processors:
            raise ValueError(f"ProcessorGraph has no processor named '{target_name}'")

        source_processor: Processor = self.processors[source_name]
        target_processor: Processor = self.processors[target_name]

        if source_field not in source_processor.outputs():
            raise ValueError(
                f"'{source_name}: {type(source_processor).__name__}' has no output named '{source_name}'. "
                f"Options are: {source_processor.outputs()}"
            )
        if target_field not in target_processor.inputs():
            raise ValueError(
                f"'{target_name}: {type(target_processor).__name__}' has no input named '{target_name}'. "
                f"Options are: {target_processor.inputs()}"
            )

        out_dim, in_dim = (
            source_processor.out_dims[source_field],
            target_processor.in_dims[target_field],
        )

        if out_dim != in_dim:
            raise ValueError(
                f"Incompatible shapes: "
                f"{out_dim}-dimensional {source_name}.{source_field}"
                " --/--> "
                f"{in_dim}-dimensional {target_name}.{target_field}"
            )

        self.connections[(source_name, source_field)] = (target_name, target_field)
        source_processor.connect(target_processor, source_field, target_field)

    def forward(self, inputs: TensorDict) -> TensorDict:
        def compute(processor: Processor):
            # first ensure all incoming fields have been computed
            for source_field, source_processor in processor.incoming:
                if source_field not in inputs:
                    compute(source_processor)

            # then compute this processor's outputs
            inputs.update(processor.forward(inputs))

        for processor in self.processors.values():
            compute(processor)

        return inputs


class Sequential(Processor):
    def __init__(self, processors: List[Processor]):
        super().__init__()
        self.processors: List[Processor] = ModuleList(processors)  # type: ignore

    def in_dims(self) -> Dict[str, Tuple[int, ...]]:
        return dict(i for dct in [proc.in_dims for proc in self.processors] for i in dct.items())

    def out_dims(self) -> Dict[str, Tuple[int, ...]]:
        return self.processors[-1].out_dims

    def connect(self, target: Processor, source_field: str, target_field: str):
        # when an output field of one processor is connected elsewhere, connect up all internal matching fields
        for proc_a in self.processors:
            for proc_b in self.processors:
                for field in proc_a.inputs():
                    if field in proc_b.outputs():
                        proc_a.connect(proc_b, field, field)
        return super().connect(target, source_field, target_field)

    def forward(self, inputs: TensorDict) -> TensorDict:
        for processor in self.processors:
            inputs = processor(inputs)
        return inputs


class StyleGAN2Mapping(Processor):
    def __init__(self, ckpt_path: str):
        super().__init__()
        self.model = StyleGAN2MappingNetwork(ckpt_path)

    def in_dims(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "z": (self.model.z_dim,),
            "c": (self.model.c_dim,),
            "truncation": (1,),
        }

    def out_dims(self) -> Dict[str, Tuple[int, ...]]:
        return {"w": (self.model.n_layers, self.model.w_dim)}

    def forward(self, inputs: TensorDict) -> TensorDict:
        w = self.model(z=inputs["z"], c=inputs["c"], truncation=inputs["truncation"])
        return TensorDict({"w": w})


class StyleGAN2Synthesis(Processor):
    def __init__(self, ckpt_path: str):
        super().__init__()
        self.model = StyleGAN2SynthesisNetwork(ckpt_path)

    def in_dims(self) -> Dict[str, Tuple[int, ...]]:
        return {"w": (self.model.n_layers, self.model.w_dim)}

    def out_dims(self) -> Dict[str, Tuple[int, ...]]:
        return {"img": (3, self.model.resolution, self.model.resolution)}

    def forward(self, inputs: TensorDict) -> TensorDict:
        return self.model(w=inputs["w"])


class StyleGAN2(Sequential):
    def __init__(self, ckpt_path: str):
        super().__init__([StyleGAN2Mapping(ckpt_path), StyleGAN2Synthesis(ckpt_path)])
