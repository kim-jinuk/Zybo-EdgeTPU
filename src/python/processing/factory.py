# processing/factory.py
from .enhancers import GammaContrast, UnsharpMask, Compose

def build_preprocessing(cfg: dict):
    if not cfg:
        return lambda x: x  # identity

    steps = []
    if cfg.get("gamma", {}).get("enable", False):
        gamma = cfg["gamma"].get("value", 1.5)
        steps.append(GammaContrast(gamma))
    if cfg.get("unsharp", {}).get("enable", False):
        strength = cfg["unsharp"].get("value", 1.0)
        steps.append(UnsharpMask(strength))

    return Compose(steps) if steps else lambda x: x