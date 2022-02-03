import os

import numpy as np
import torch
from maua.GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._set_cublas_allow_tf32(True)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.jit.optimized_execution(True)
torch.jit.fuser("fuser2")


def GlumpyWindow(module, start_state, w, h):
    from contextlib import contextmanager

    import pycuda.driver
    from glumpy import app, gl, gloo
    from pycuda.gl import graphics_map_flags

    VERTEX = """
    uniform float scale;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        v_texcoord = texcoord;
        gl_Position = vec4(scale*position, 0.0, 1.0);
    } """

    FRAGMENT = """
    uniform sampler2D tex;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(tex, v_texcoord);
    } """

    @contextmanager
    def cuda_activate(img):
        """Context manager simplifying use of pycuda.gl.RegisteredImage"""
        mapping = img.map()
        yield mapping.array(0, 0)
        mapping.unmap()

    app.use("pyglet")
    window = app.Window(w, h, fullscreen=True)

    global state
    state = start_state

    @window.event
    def on_draw(dt):
        global state

        state, img = module(state)
        img = img.squeeze().permute(1, 2, 0).mul(255).byte().data
        img = torch.cat((img, img[:, :, [0]]), 2)
        img[:, :, 3] = 255
        img = img.contiguous()

        tex = screen["tex"]
        h, w = tex.shape[:2]
        img = img[:h, :w]
        if tuple(img.shape) != tuple(tex.shape):
            texture = torch.zeros(tex.shape, device=img.device, dtype=img.dtype)
            th, tw = img.shape[:2]
            texture[:th, :tw] = img[:th, :tw]
        else:
            texture = img

        assert tex.nbytes == texture.numel() * texture.element_size()
        with cuda_activate(cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(texture.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes // h
            cpy.height = h
            cpy(aligned=False)
            torch.cuda.synchronize()

        window.clear()
        screen.draw(gl.GL_TRIANGLE_STRIP)

    w, h = window.get_size()

    import pycuda.gl
    import pycuda.gl.autoinit

    tex = np.zeros((h, w, 4), np.uint8).view(gloo.Texture2D)
    tex.activate()  # force gloo to create on GPU
    tex.deactivate()
    cuda_buffer = pycuda.gl.RegisteredImage(int(tex.handle), tex.target, graphics_map_flags.WRITE_DISCARD)
    screen = gloo.Program(VERTEX, FRAGMENT, count=4)
    screen["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    screen["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    screen["scale"] = 1.0
    screen["tex"] = tex

    app.run()


class RealtimeModule(torch.nn.Module):
    __constants__ = ["motion_react", "motion_randomness", "motion_smooth", "truncation"]

    def __init__(
        self,
        model_file,
        w,
        h,
        motion_react,
        motion_randomness,
        motion_smooth,
        truncation,
        resize_strategy,
        resize_layer,
        device,
        dtype,
    ):
        super().__init__()

        self.motion_react, self.motion_randomness, self.motion_smooth = motion_react, motion_randomness, motion_smooth
        self.truncation = truncation

        self.G_map = StyleGAN2Mapper(model_file).to(device)
        self.G_synth = StyleGAN2Synthesizer(model_file, (w, h), strategy=resize_strategy, layer=resize_layer).to(device)

        rand_factors = torch.ones(B, 512, dtype=dtype, device=device)
        rand_factors[torch.rand_like(rand_factors) > 0.5] -= 0.5
        self.register_buffer("rand_factors", rand_factors)
        self.register_buffer("motion_signs", torch.sign(torch.randn(B, 512, dtype=dtype, device=device)))

        self.i = torch.tensor(0)

    def forward(self, latent: torch.Tensor):
        # update motion directions to stay in decent latent space
        self.motion_signs[latent - self.motion_react < -2 * self.truncation] *= -1
        self.motion_signs[latent + self.motion_react >= 2 * self.truncation] *= -1

        # Re-initialize randomness factors every 4 seconds
        if self.i % (24 * 4) == 0:
            new_factors = torch.ones_like(self.rand_factors)
            new_factors[torch.rand_like(self.rand_factors) > 0.5] -= 0.5
            self.rand_factors.set_(new_factors.data)

        motion_noise = self.motion_react * self.motion_signs * self.rand_factors

        latent = latent * self.motion_smooth + (latent + motion_noise) * (1 - self.motion_smooth)

        self.i.set_(self.i + 1)

        mapped_latent = self.G_map(latent)
        raw_img = self.G_synth(mapped_latent)
        img = raw_img.add(1).div(2).clamp(0, 1)
        return latent, img


def fx(module):
    from torch.fx import symbolic_trace
    from torch.fx.experimental.fx2trt.fx2trt import TRTInterpreter, InputTensorSpec, TRTModule

    return TRTModule(
        *TRTInterpreter(
            symbolic_trace(module),
            InputTensorSpec(shape=latent.shape, dtype=latent.dtype, device=latent.device),
        ).run(max_batch_size=1)
    )


def mobile(module):
    from torch.utils import mobile_optimizer

    return mobile_optimizer.optimize_for_mobile(module)


def onnxrt(module):
    import onnxruntime as ort

    torch.onnx.export(
        module,
        torch.randn((B, 512), device=device, dtype=dtype),
        "generator.onnx",
        export_params=True,
        verbose=True,
        opset_version=11,
    )

    ort_session = ort.InferenceSession("generator.onnx")

    def generate(x):
        ort_inputs = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs

    return generate


def tvm(module, use_onnx=True):
    import tvm
    import tvm.contrib.graph_executor as runtime
    import tvm.relay as relay

    if use_onnx:
        import onnx

        torch.onnx.export(
            next_frame,
            torch.randn((B, 512), device=device, dtype=dtype),
            "generator.onnx",
            export_params=True,
            verbose=True,
            opset_version=11,
        )
        mod, params = relay.frontend.from_onnx(onnx.load("generator.onnx"), shape={"0": (B, 512)})
    else:
        mod, params = relay.frontend.from_pytorch(module, [("input", (B, 512))])
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=device, params=params)
    m = runtime.GraphModule(lib["default"](tvm.device(device)))

    def next_frame_tvm(input):
        m.set_input("input", tvm.nd.array(input))
        m.run()
        return (m.get_output(0), m.get_output(1))

    return next_frame_tvm


def tensorrt(module, shape):
    import torch_tensorrt as tt

    tt.logging.set_reportable_log_level(tt.logging.Level.Error)
    module_trt = tt.compile(
        module,
        inputs=[tt.Input(shape, dtype=dtype)],
        enabled_precisions={dtype},
    )
    print("success?")
    return module_trt


if __name__ == "__main__":
    with torch.inference_mode():
        B = 1
        w, h = 1920, 1024
        model_file = None  # "/home/hans/modelzoo/wavefunk/diffuseSWA/diffuse-gamma1e-4-001000_diffuse-gamma1e-4-007500_diffuse-gamma1e-6-001000_diffuse-007500_bad_diffuse-gamma1e-4-001500_diffuse-gamma1e-5-006000_diffus-1024-randomSWA.pt"
        motion_react = 0.5
        motion_randomness = 0.5
        motion_smooth = 0.75
        truncation = 1
        resize_strategy = "pad-reflect-out"
        resize_layer = 2
        device = "cuda"
        dtype = torch.float32
        latent = torch.randn(B, 512, dtype=dtype, device=device)

        next_frame = RealtimeModule(
            model_file,
            w,
            h,
            motion_react,
            motion_randomness,
            motion_smooth,
            truncation,
            resize_strategy,
            resize_layer,
            device,
            dtype,
        )
        next_frame = next_frame.eval().to(device)
        next_frame(latent)

        import torch2trt
        import tensorrt as trt

        # next_frame_trace = torch.jit.trace(next_frame, example_inputs=latent, check_trace=False)

        # next_frame.G_synth = torch.jit.script(next_frame.G_synth)
        # [print(v["method_str"]) for v in torch2trt.CONVERTERS.values()]
        next_frame_trt = torch2trt.torch2trt(
            next_frame.G_synth,
            [torch.randn(B, 18, 512, device=device, dtype=dtype)],
            fp16_mode=True,
            log_level=trt.Logger.VERBOSE,
        )
        exit(0)
        # next_frame.G_synth = torch.jit.optimize_for_inference(next_frame.G_synth)
        next_frame.G_synth = tensorrt(next_frame.G_synth, shape=(B, 18, 512))

        next_frame.G_map = torch.jit.script(next_frame.G_map)
        # next_frame.G_map = torch.jit.optimize_for_inference(next_frame.G_map)
        next_frame.G_map = tensorrt(next_frame.G_map, shape=(B, 512))

        print(next_frame)

        GlumpyWindow(next_frame, latent, w, h)
