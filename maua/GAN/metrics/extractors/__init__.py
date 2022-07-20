def get_extractor(name):
    if name.lower() == "swav":
        from .swav import SwAV

        return SwAV(), 224
    elif name.lower() == "inception":
        from .inception import Inception

        return Inception(), 299
    else:
        raise ValueError(f"Unknown extractor {name}")
