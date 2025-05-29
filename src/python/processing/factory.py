# processing/factory.py
from .enhancers import LightCLAHE, GammaContrast, UnsharpMask, Compose

def build_preprocessing(cfg: dict):
    if not cfg:
        return lambda x: x  # identity

    steps = []
    if cfg.get("Contrast", {}).get("enable", False):
        clip_limit = cfg["Contrast"].get("value1", 1.5)
        tile = cfg["Contrast"].get("value2", 4)
        steps.append(LightCLAHE(clip_limit, tile))

    if cfg.get("EdgeEnhance", {}).get("enable", False):
        strength = cfg["EdgeEnhance"].get("value", 1.0)
        steps.append(UnsharpMask(strength))

    if cfg.get("Denoise", {}).get("enable", False):
        ksize = cfg["Denoise"].get("value1", 3)
        sigma = cfg["Denoise"].get("value2", 0.0)
        steps.append(UnsharpMask(ksize, sigma))

    return Compose(steps) if steps else lambda x: x