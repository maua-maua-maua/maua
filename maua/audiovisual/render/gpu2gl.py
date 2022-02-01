from contextlib import contextmanager

import numpy as np
import pycuda.driver
import torch
from torch._C import layout
import torch.nn.functional as F
from glumpy import app, gl, gloo
from maua.GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from pycuda.gl import graphics_map_flags

torch._C._set_cublas_allow_tf32(True)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.jit.optimized_execution(True)
torch.jit.fuser("fuser2")


@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()


def create_shared_texture(w, h, c=4, map_flags=graphics_map_flags.WRITE_DISCARD, dtype=np.uint8):
    """Create and return a Texture2D with gloo and pycuda views."""
    print(h, w, c)
    tex = np.zeros((h, w, c), dtype).view(gloo.Texture2D)
    tex.activate()  # force gloo to create on GPU
    tex.deactivate()
    cuda_buffer = pycuda.gl.RegisteredImage(int(tex.handle), tex.target, map_flags)
    return tex, cuda_buffer


def setup():
    global screen, cuda_buffer, state
    w, h = window.get_size()
    # setup pycuda and torch
    print("init pycuda")
    import pycuda.gl
    import pycuda.gl.autoinit

    assert torch.cuda.is_available()
    print("using GPU {}".format(torch.cuda.current_device()))
    # torch.nn layers expect batch_size, channels, height, width
    state = torch.cuda.FloatTensor(1, 3, h, w)
    state.uniform_()
    # create a buffer with pycuda and gloo views
    tex, cuda_buffer = create_shared_texture(w, h, 4)
    # create a shader to program to draw to the screen
    vertex = """
    uniform float scale;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        v_texcoord = texcoord;
        gl_Position = vec4(scale*position, 0.0, 1.0);
    } """
    fragment = """
    uniform sampler2D tex;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(tex, v_texcoord);
    } """
    # Build the program and corresponding buffers (with 4 vertices)
    screen = gloo.Program(vertex, fragment, count=4)
    # Upload data into GPU
    screen["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    screen["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    screen["scale"] = 1.0
    screen["tex"] = tex


model_file = "/home/hans/modelzoo/wavefunk/diffuseSWA/diffuse-gamma1e-4-001000_diffuse-gamma1e-4-007500_diffuse-gamma1e-6-001000_diffuse-007500_bad_diffuse-gamma1e-4-001500_diffuse-gamma1e-5-006000_diffus-1024-randomSWA.pt"
G_map = StyleGAN2Mapper(model_file).cuda()
w, h = 1920, 1024
G_synth = StyleGAN2Synthesizer(model_file, (w, h), "pad-reflect-out", 2).cuda()
motion_react = 0.5
motion_randomness = 0.5
motion_smooth = 0.75
truncation = 1
latent = torch.randn(1, 512, device="cuda")
motion_signs = torch.sign(torch.randn_like(latent))
rand_factors = 1 - motion_randomness * torch.rand_like(latent)
rand_factors[torch.rand_like(rand_factors) > 0.5] = 1
i = 0


def torch_process(state):
    global latent, rand_factors, i

    # update motion directions to stay in decent latent space
    motion_signs[latent - motion_react < -2 * truncation] = 1
    motion_signs[latent + motion_react >= 2 * truncation] = -1

    # Re-initialize randomness factors every 4 seconds
    if i % round(max(24, window.fps) * 4) == 0:
        rand_factors = 1 - motion_randomness * torch.rand_like(latent)
        rand_factors[torch.rand_like(rand_factors) > 0.5] = 1

    motion_noise = motion_react * motion_signs * rand_factors

    latent = latent * motion_smooth + (latent + motion_noise) * (1 - motion_smooth)

    i += 1
    return G_synth(latent_w_plus=G_map(latent)).add(1).div(2).clamp(0, 1)


# create window with OpenGL context
app.use("pyglet")
window = app.Window(w, h, fullscreen=False)


@window.event
def on_draw(dt):
    global state
    window.set_title(str(window.fps))
    tex = screen["tex"]
    h, w = tex.shape[:2]
    # mutate state in torch
    state = torch_process(state)  # prevent autograd from filling memory
    # convert into proper format
    tensor = state.squeeze().permute(1, 2, 0).data  # put in texture order
    tensor = torch.cat((tensor, tensor[:, :, [0]]), 2)  # add the alpha channel
    tensor[:, :, 3] = 1  # set alpha
    tensor = tensor[:h, :w]  # ensure tensor has same shape as screen size (e.g. when screen is resized to be smaller)
    if tuple(tensor.shape) != tuple(tex.shape):
        texture = torch.zeros(tex.shape, device=tensor.device, dtype=tensor.dtype)
        th, tw = tensor.shape[:2]
        texture[:th, :tw] = tensor[:th, :tw]
    else:
        texture = tensor
    # check that tensor order matches texture:
    # img[:,:,2] = 1 # set blue
    # img[100,:,:] = 1 # horizontal white line
    # img[:,200,0] = 1 # vertical magenta line
    texture = (255 * texture).byte().contiguous()  # convert to ByteTensor
    # copy from torch into buffer
    assert tex.nbytes == texture.numel() * texture.element_size()
    with cuda_activate(cuda_buffer) as ary:
        cpy = pycuda.driver.Memcpy2D()
        cpy.set_src_device(texture.data_ptr())
        cpy.set_dst_array(ary)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes // h
        cpy.height = h
        cpy(aligned=False)
        torch.cuda.synchronize()
    # draw to screen
    window.clear()
    screen.draw(gl.GL_TRIANGLE_STRIP)


# not sure why this doesn't work right
@window.event
def on_close():
    pycuda.gl.autoinit.context.pop()


if __name__ == "__main__":
    with torch.inference_mode():
        setup()
        app.run()


# class Glumpy(Renderer):
#     def __init__(self, output_file, fps=24, audio_file=None, audio_offset=0, audio_duration=None, ffmpeg_preset="slow"):
#         super().__init__()
#         self.output_file, self.fps, self.ffmpeg_preset = output_file, fps, ffmpeg_preset
#         self.audio_file, self.audio_offset, self.audio_duration = audio_file, audio_offset, audio_duration

#     def __call__(self, synthesizer, inputs, postprocess, fp16=True):
#         dataset = TensorDataset(*inputs.values())

#         def collate_fn(data):
#             return {
#                 k: v.unsqueeze(0).to(self.device, dtype=torch.float16 if fp16 else torch.float32)
#                 for k, v in zip(inputs.keys(), data[0])
#             }

#         loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
#         synthesizer = synthesizer.to(self.device)

#         if fp16:

#             def force_half(mod):
#                 if hasattr(mod, "use_fp16"):
#                     mod.use_fp16 = True
#                 if hasattr(mod, "noise_const"):
#                     setattr(mod, "noise_const", mod.noise_const.half())

#             synthesizer.G_synth.apply(force_half)

#         with VideoWriter(
#             self.output_file,
#             synthesizer.output_size,
#             self.fps,
#             self.audio_file,
#             self.audio_offset,
#             self.audio_duration,
#             self.ffmpeg_preset,
#         ) as video:
#             for batch in tqdm(loader):
#                 frame = synthesizer(**batch).add(1).div(2)
#                 frame = postprocess(frame)
#                 video.write(frame)

#         return VideoReader(self.output_file)
