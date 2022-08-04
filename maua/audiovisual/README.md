# Maua Audiovisual

This module contains the follow-up implementation to [maua-stylegan2](https://github.com/JCBrouwer/maua-stylegan2). This supports audio-reactive video synthesis, originally described [in this blog post](https://wavefunk.xyz/audio-reactive-stylegan) as well as in [this master thesis (PDF)](https://jcbrouwer.github.io/thesis/Hans%20Brouwer%20-%20Self-supervised%20Audio-reactive%20Music%20Video%20Synthesis.pdf).

Right now StyleGAN2 and StyleGAN3 are supported, but more models will be added.

The code can be run like this (from the root of the repo):
```bash
python -m maua.audiovisual.generate --audio_file /path/to/audio.wav --model_file /path/to/sg3.pt --patch_file maua/audiovisual/patches/examples/stylegan3.py
```

To change the audio-reactivity you can edit `maua/audiovisual/patches/examples/stylegan3.py` (or make a copy somewhere else and change --patch_file). Tutorials will be made available by Maua version 1.0.
