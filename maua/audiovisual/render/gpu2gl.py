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
        img = img[0].squeeze().permute(1, 2, 0).mul(255).byte().data
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
        inference_network=False,
    ):
        super().__init__()

        self.motion_react, self.motion_randomness, self.motion_smooth = motion_react, motion_randomness, motion_smooth
        self.truncation = truncation

        self.G_map = StyleGAN2Mapper(model_file, inference_network).to(device)
        self.G_synth = StyleGAN2Synthesizer(
            model_file, inference_network, (w, h), strategy=resize_strategy, layer=resize_layer
        ).to(device)

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


if __name__ == "__main__":
    with torch.inference_mode():
        B = 3
        w, h = 1920, 1024
        model_file = "/home/hans/modelzoo/wavefunk/diffuseSWA/diffuse-gamma1e-4-001000_diffuse-gamma1e-4-007500_diffuse-gamma1e-6-001000_diffuse-007500_bad_diffuse-gamma1e-4-001500_diffuse-gamma1e-5-006000_diffus-1024-randomSWA.pt"
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
        next_frame = next_frame.half().eval().to(device)
        for b in next_frame.G_synth.G_synth.bs:
            b.resample_filter = b.resample_filter.float()
            try:
                b.conv0.resample_filter = b.conv0.resample_filter.float()
            except:
                pass
            b.conv1.resample_filter = b.conv1.resample_filter.float()

        # next_frame = torch.jit.trace(next_frame, latents_z)
        # next_frame = torch.jit.optimize_for_inference(next_frame)
        # next_frame(latents_z)

        GlumpyWindow(next_frame, latents_z, w, h)
