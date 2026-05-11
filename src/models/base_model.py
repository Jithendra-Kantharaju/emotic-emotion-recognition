"""
models/base_model.py
====================
Abstract base class that all VLM wrappers must implement.
"""

from abc import ABC, abstractmethod
from PIL import Image


class BaseVLM(ABC):
    """Abstract wrapper for a vision-language model."""

    def __init__(self, model_key: str, config: dict):
        self.model_key  = model_key
        self.model_name = config["name"]
        self.hf_id      = config["hf_id"]
        self.model      = None
        self.processor  = None

    @abstractmethod
    def load(self):
        """Download / initialise model weights and processor."""
        ...

    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> str:
        """
        Run a single forward pass.
        Must return the raw text output from the model.
        """
        ...

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.model_name}'>"
