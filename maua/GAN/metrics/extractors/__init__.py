def get_extractor(name):
    if name.lower() == "swav":
        from maua.GAN.metrics.extractors.swav import SwAV

        return SwAV(), 224
    elif name.lower() == "inception":
        from maua.GAN.metrics.extractors.inception import Inception

        return Inception(), 299
    else:
        raise ValueError(f"Unknown extractor {name}")
