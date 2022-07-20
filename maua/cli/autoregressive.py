import argparse

from . import main_function


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser("ru", parents=[ru_dalle()], help="Generate images with RuDALL-E", add_help=False)
    subparsers.add_parser("min", parents=[min_dalle()], help="Generate images with MinDALL-E", add_help=False)
    subparsers.add_parser("rq", parents=[rq_dalle()], help="Generate images with RQVAE Transformer", add_help=False)
    return parser


def ru_dalle():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "generate", parents=[ru_dalle_generate()], help="Generate images with RuDALL-E", add_help=False
    ).set_defaults(func="maua.autoregressive.ru_dalle.generate")
    subparsers.add_parser(
        "finetune",
        parents=[ru_dalle_finetune()],
        help="Finetune RuDALL-E on a set of images (and captions)",
        add_help=False,
    ).set_defaults(func="maua.autoregressive.ru_dalle.finetune")
    subparsers.add_parser(
        "api",
        parents=[ru_dalle_api()],
        help="Request RuDALL-E Kandinsky images from the Sbercloud API",
        add_help=False,
    ).set_defaults(func="maua.autoregressive.ru_dalle.api")
    return parser


def ru_dalle_generate():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str, default="", help="Input text to sample images for after training. Will only have effect if you train with low number of steps / low learning rate or train with captions.")
    parser.add_argument("--num_outputs", type=int, default=8, help="Number of images to generate after finetuning")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to sample at once. Higher batch size requires more memory, but will be faster per sample overall. Inference batches can be bigger as we don't need to store gradients for training.")
    parser.add_argument("--size", type=str, default='256,256', help="width,height of images to sample")
    parser.add_argument("--stretch_size", type=str, default=None, help="width,height to stretch sampled images to (will only give decent results if model was finetuned with this stretch size).")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2, 4, 8], help="Use RealESRGAN to upscale outputs.")
    parser.add_argument("--top_p", type=float, default=0.99, help="Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to train on, using 'cpu' will take a long time!")
    parser.add_argument("--low_memory", action="store_true", help="Enable if you have less than 16 GB of (V)RAM to use gradient checkpointing (slower but more memory efficient)")
    parser.add_argument("--no_oversample", action="store_true", help="Disable oversampling procedure. Oversampling is slower but works better when sampling shapes different from what the model was trained on.")
    parser.add_argument("--checkpoint", type=str, default=None, help=f"Checkpoint to resume from. Either a path to a trained RuDALL-E checkpoint or see the list in --model-help.")  # TODO --model-help
    parser.add_argument("--output_name", type=str, default=None, help="Name to save images under.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save output images in.")
    # fmt: on
    return parser


def ru_dalle_finetune():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with images to train on")
    parser.add_argument("--images", type=list, default=[], nargs="*", help="List of images to train on")
    parser.add_argument("--captions", type=list, default=[], nargs="*", help="List of a caption for each image")
    parser.add_argument("--input_text", type=str, default="", help="Input text to sample images for after training. Will only have effect if you train with low number of steps / low learning rate or train with captions.")
    parser.add_argument("--num_outputs", type=int, default=8, help="Number of images to generate after finetuning")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of example images from input_dir to train on (None => All images will be used)")
    parser.add_argument("--steps", type=int, default=500, help="Number of batches to finetune for. More steps will converge to outputs that are closer to the inputs (which also means less variation).")
    parser.add_argument("--lr", type=float, default=1e-5, help="Starting learning rate (decays by a factor of 500 over training). A high learning rate will converge faster (which also means less variation). 1e-4 to 1e-5 is a good starting range (with 1e-5 resulting in more varied outputs).")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of images for each training step. Higher batch size requires more memory, but will be faster per sample overall.")
    parser.add_argument("--inference_batch_size", type=int, default=1, help="Number of images to sample at once. Higher batch size requires more memory, but will be faster per sample overall. Inference batches can be bigger as we don't need to store gradients for training.")
    parser.add_argument("--size", type=str, default='256,256', help="width,height of images to generate")
    parser.add_argument("--stretch", action="store_true", help="Squash images down to fixed size for training and stretch back to original size after sampling. This can significantly improve training/sampling time while still yielding good results. All training images should have the same aspect ratio.")
    parser.add_argument("--random_crop", type=int, default=None, help="Randomly crop sections of this size during training. None disables random cropping.")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2, 4, 8], help="Use RealESRGAN to upscale outputs.")
    parser.add_argument("--top_p", type=float, default=0.99, help="Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to train on, using 'cpu' will take a long time!")
    parser.add_argument("--low_memory", action="store_true", help="Enable if you have less than 16 GB of (V)RAM to use gradient checkpointing (slower but more memory efficient)")
    parser.add_argument("--adam8bit", action="store_true", help="Enable for even more memory-efficient training.")
    parser.add_argument("--checkpoint", type=str, default=None, help=f"Checkpoint to resume from. Either a path to a trained RuDALL-E checkpoint see the list in --model-help.")  # TODO --model-help
    parser.add_argument("--save_dir", type=str, default="modelzoo/", help="Directory to save finetuned checkpoints in.")
    parser.add_argument("--model_name", type=str, default=None, help="Name for finetuned checkpoints. Will default to the name of input_dir or the first input_img.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save output images in.")
    # fmt: on
    return parser


def ru_dalle_api():
    # fmt:off
    parser = argparse.ArgumentParser()
    parser.add_argument("input_text", help="Text for which an image should be generated (in English, will be translated to Russian)")
    parser.add_argument("request_url", help="API URL of the Sbercloud RuDALL-E Kandinsky deployment")
    parser.add_argument("--top_k", type=int, default=1500)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--images_num", type=int, default=4)
    parser.add_argument("--rerank_top", type=int, default=4)
    parser.add_argument("--out_dir", default='output/')
    parser.add_argument("--verbose", action='store_true')
    # fmt:on
    return parser


def dayma_dalle():
    pass  # TODO


def min_dalle():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "generate", parents=[min_dalle_generate()], help="Generate images with MinDALL-E", add_help=False
    ).set_defaults(func=main_function("maua.autoregressive.min_dalle.generate"))
    # subparsers.add_parser(
    #     "finetune", parents=[min_dalle_finetune()], help="Fine-tune MinDALL-E", add_help=False
    # ).set_defaults(func=main_function('maua.autoregressive.min_dalle.finetune'))
    return parser


def min_dalle_generate():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Input text to sample images.")
    parser.add_argument("--num_candidates", type=int, default=32, help="Number of images to generate in total")
    parser.add_argument("--num_outputs", type=int, default=8, help="Number of images to output based on best CLIP scores")
    parser.add_argument("--top_k", type=float, default=256, help="Should probably be set no higher than 256.")
    parser.add_argument("--top_p", type=float, default=None, help="Effects how closely sampled images match training data. Lower values might give higher quality images at the cost of variation. A good range is between 0.9 and 0.999.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to train on, using 'cpu' will take a long time!")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save output images in.")
    # fmt: on
    return parser


def min_dalle_finetune():
    pass  # TODO


def rq_dalle():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_prompts", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--sampling_ratio", type=int, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--checkpoint_dir", type=str, default="modelzoo/rqvae_cc3m_cc12m_yfcc")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--make_grid", action="store_true")
    parser.add_argument("--out_dir", type=str, default="output/")
    parser.set_defaults(func=main_function("maua.autoregressive.rq_dalle"))
    return parser
