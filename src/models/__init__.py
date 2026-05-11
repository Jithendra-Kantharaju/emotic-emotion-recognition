"""
models/__init__.py
==================
Factory function — returns an instantiated (but not yet loaded) VLM.
Call .load() on the result to download and initialise weights.
"""

from src.config import MODELS
from src.models.base_model      import BaseVLM
from src.models.qwen_model      import QwenVLModel
from src.models.llava_model     import LLaVAModel
from src.models.internvl_model  import InternVLModel

_LOADER_MAP = {
    "qwen":     QwenVLModel,
    "llava":    LLaVAModel,
    "internvl": InternVLModel,
}


def get_model(model_key: str) -> BaseVLM:
    if model_key not in MODELS:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Choose from: {list(MODELS.keys())}"
        )
    cfg = MODELS[model_key]
    cls = _LOADER_MAP[cfg["loader"]]
    return cls(model_key=model_key, config=cfg)
