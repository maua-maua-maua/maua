from math import sqrt
import os

import cloudpickle
import numpy as np
import torch
from maua.GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
import tensorrt as trt
import torch2trt

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
        self.motion_signs[latent - self.motion_react < -2 * self.truncation] = 1
        self.motion_signs[latent + self.motion_react >= 2 * self.truncation] = -1

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


@torch2trt.tensorrt_converter("torch.nn.functional.conv_transpose2d")
def convert_conv_transpose2d(ctx):
    input = torch2trt.get_arg(ctx, "input", pos=0, default=None)
    weight = torch2trt.get_arg(ctx, "weight", pos=1, default=None)
    bias = torch2trt.get_arg(ctx, "bias", pos=2, default=None)
    stride = torch2trt.get_arg(ctx, "stride", pos=3, default=1)
    padding = torch2trt.get_arg(ctx, "padding", pos=4, default=0)
    padding = torch2trt.get_arg(ctx, "output_padding", pos=5, default=0)
    groups = torch2trt.get_arg(ctx, "groups", pos=6, default=1)
    dilation = torch2trt.get_arg(ctx, "dilation", pos=7, default=1)
    input_trt = torch2trt.add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_dim = input.dim() - 2

    out_channels = int(weight.shape[1])

    kernel_size = tuple(weight.shape[2:])
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size,) * input_dim

    if not isinstance(stride, tuple):
        stride = (stride,) * input_dim

    if not isinstance(padding, tuple):
        padding = (padding,) * input_dim

    if not isinstance(dilation, tuple):
        dilation = (dilation,) * input_dim

    kernel = weight.detach().cpu().numpy()

    if bias is not None:
        bias = bias.detach().cpu().numpy()

    layer = ctx.network.add_deconvolution_nd(
        input=input_trt,
        num_output_maps=out_channels * groups,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias,
    )
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    if groups is not None:
        layer.num_groups = groups

    output._trt = layer.get_output(0)


@torch2trt.tensorrt_converter("torch.zeros")
@torch2trt.tensorrt_converter("torch.ones")
@torch2trt.tensorrt_converter("torch.randn")
@torch2trt.tensorrt_converter("torch.randn_like")
def convert_as_constant(ctx):
    output = ctx.method_return
    layer = ctx.network.add_constant(tuple(output.shape), output.detach().cpu().numpy())
    output._trt = layer.get_output(0)


if __name__ == "__main__":
    with torch.inference_mode():

        class SimpleModConv(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lw = torch.nn.Parameter(torch.randn(512, 512))
                self.lb = torch.nn.Parameter(torch.randn(512))
                self.w = torch.nn.Parameter(torch.randn(512, 512, 3, 3))
                self.gain = sqrt(2)

            def forward(self, x, z):
                B, xc, xh, xw = x.shape
                wco, wci, kh, kw = self.w.shape

                # normalize w by input elements and L-infinity norm
                numin = sqrt(wci * kh * kw)
                linf = torch.max(torch.max(torch.max(torch.abs(self.w), dim=1).values, dim=1).values, dim=1).values
                w = self.w / (numin * linf).reshape(wco, 1, 1, 1)

                # affine transform of latent vector
                styles = torch.nn.functional.linear(z, self.gain * self.lw, self.gain * self.lb)

                # modulate weight by style per instance
                w = w.reshape(1, wco, wci, kh, kw) * styles.reshape(B, 1, wci, 1, 1)

                # normalize weights to ensure unit scaling
                w = w / ((w * w).sum((2, 3, 4)) + 1e-8).sqrt().reshape(B, wco, 1, 1, 1)

                # reshape and perform convolution with separate weights per instance (fused in one op by groups)
                x = x.reshape(1, B * xc, xh, xw)
                w = w.reshape(B * wco, wci, kh, kw)
                x = torch.nn.functional.conv2d(x, w, padding=(1, 1), groups=B)
                x = x.reshape(B, wco, xh, xw)

                # add a little noise
                return x + torch.randn_like(x)

        module = SimpleModConv().cuda()
        simple_mod_conv = torch2trt.torch2trt(
            module=module,
            inputs=[torch.randn(3, 512, 128, 128).cuda(), torch.randn(3, 512).cuda()],
            input_names=["x", "z"],
            output_names=["y"],
            log_level=trt.Logger.INFO,
            max_batch_size=3,
            fp16_mode=True,
            max_workspace_size=1 << 33,
        )

        x, z = torch.randn(3, 512, 128, 128).cuda(), torch.randn(3, 512).cuda()
        xth = module(x, z)
        xtrt = simple_mod_conv(x, z)

        print(torch.abs(xtrt - xth).sum())
        assert torch.allclose(xtrt, xth)

        exit()
        B = 3
        w, h = 1920, 1024
        model_file = None  # "/home/hans/modelzoo/wavefunk/diffuseSWA/diffuse-gamma1e-4-001000_diffuse-gamma1e-4-007500_diffuse-gamma1e-6-001000_diffuse-007500_bad_diffuse-gamma1e-4-001500_diffuse-gamma1e-5-006000_diffus-1024-randomSWA.pt"
        motion_react = 0.5
        motion_randomness = 0.5
        motion_smooth = 0.75
        truncation = 1
        resize_strategy = "stretch"
        resize_layer = 6
        device = "cuda"
        dtype = torch.half
        latents_z = torch.randn(B, 512, dtype=dtype, device=device)
        latents_w = torch.randn(B, 18, 512, device=device, dtype=dtype)

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
        next_frame = next_frame.eval().half().to(device)
        next_frame(latents_z)

        next_frame.G_map = torch2trt.torch2trt(
            module=next_frame.G_map,
            inputs=[latents_z],
            input_names=["latents_z"],
            output_names=["latents_w"],
            log_level=trt.Logger.INFO,
            max_batch_size=B,
            fp16_mode=True,
            max_workspace_size=1 << 33,
        )
        print(next_frame.G_map(torch.randn(B, 512, device=device, dtype=dtype)).shape)

        next_frame.G_synth = torch2trt.torch2trt(
            module=next_frame.G_synth,
            inputs=[latents_w],
            input_names=["latents_w"],
            output_names=["images"],
            log_level=trt.Logger.INFO,
            max_batch_size=B,
            fp16_mode=True,
            max_workspace_size=1 << 33,
        )
        print(next_frame.G_synth(torch.randn(B, 18, 512, device=device, dtype=dtype)).shape)

        next_frame(latents_z)

        GlumpyWindow(next_frame, latents_z, w, h)
