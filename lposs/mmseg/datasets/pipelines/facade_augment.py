"""Custom augmentation operations tailored to facade imagery."""

import random
from typing import Dict

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from ..builder import PIPELINES


@PIPELINES.register_module()
class FacadeAugment:
    """Apply colour jitter and random erosion to mimic facade degradations."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, blur_prob: float = 0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.blur_prob = blur_prob

    def _jitter(self, img: np.ndarray) -> np.ndarray:
        delta = (random.random() * 2 - 1) * self.brightness * 255
        img = img.astype(np.float32) + delta
        factor = 1.0 + (random.random() * 2 - 1) * self.contrast
        img = (img - 127.5) * factor + 127.5
        return np.clip(img, 0, 255).astype(np.uint8)

    def _blur(self, img: np.ndarray) -> np.ndarray:
        if cv2 is not None and random.random() < self.blur_prob:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def __call__(self, results: Dict) -> Dict:
        img = results['img']
        img = self._jitter(img)
        img = self._blur(img)
        results['img'] = img
        return results

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(brightness={self.brightness}, "
                f"contrast={self.contrast}, blur_prob={self.blur_prob})")
