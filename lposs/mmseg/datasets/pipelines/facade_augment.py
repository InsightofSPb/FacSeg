"""Custom augmentation operations tailored to facade imagery."""

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from ..builder import PIPELINES


@PIPELINES.register_module()
class FacadeAugment:
    """Apply rich colour and geometric augmentations for facade imagery."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.0,
        hue: float = 0.0,
        gamma: float = 0.0,
        blur_prob: float = 0.3,
        noise_std: float = 0.0,
        perspective_prob: float = 0.0,
        perspective_scale: float = 0.05,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.blur_prob = blur_prob
        self.noise_std = noise_std
        self.perspective_prob = perspective_prob
        self.perspective_scale = perspective_scale

    def _jitter(self, img: np.ndarray) -> np.ndarray:
        if self.brightness > 0:
            delta = (random.random() * 2 - 1) * self.brightness * 255
            img = img.astype(np.float32) + delta
        if self.contrast > 0:
            factor = 1.0 + (random.random() * 2 - 1) * self.contrast
            img = (img - 127.5) * factor + 127.5
        return np.clip(img, 0, 255).astype(np.uint8)

    def _saturation_hue(self, img: np.ndarray) -> np.ndarray:
        if cv2 is None or (self.saturation <= 0 and self.hue <= 0):
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        if self.saturation > 0:
            sat_scale = 1.0 + (random.random() * 2 - 1) * self.saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        if self.hue > 0:
            hue_delta = (random.random() * 2 - 1) * self.hue * 180
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _gamma(self, img: np.ndarray) -> np.ndarray:
        if self.gamma <= 0:
            return img
        gamma_value = 1.0 + (random.random() * 2 - 1) * self.gamma
        gamma_value = max(gamma_value, 1e-3)
        img_norm = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
        img_norm = np.power(img_norm, gamma_value)
        return np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)

    def _noise(self, img: np.ndarray) -> np.ndarray:
        if self.noise_std <= 0:
            return img
        noise = np.random.normal(0.0, self.noise_std * 255, img.shape).astype(np.float32)
        img = img.astype(np.float32) + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def _blur(self, img: np.ndarray) -> np.ndarray:
        if cv2 is not None and random.random() < self.blur_prob:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def _perspective(self, img: np.ndarray, segs: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        if cv2 is None or random.random() >= self.perspective_prob:
            return img, list(segs)
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        jitter = np.random.uniform(-self.perspective_scale, self.perspective_scale, size=(4, 2))
        jitter[:, 0] *= w
        jitter[:, 1] *= h
        dst = src + jitter
        dst[:, 0] = np.clip(dst[:, 0], 0, w - 1)
        dst[:, 1] = np.clip(dst[:, 1], 0, h - 1)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        warped_segs: List[np.ndarray] = []
        for seg in segs:
            warped = cv2.warpPerspective(
                seg,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            warped_segs.append(warped)
        return warped_img, warped_segs

    def __call__(self, results: Dict) -> Dict:
        img = results['img']
        img = self._jitter(img)
        img = self._saturation_hue(img)
        img = self._gamma(img)
        img = self._noise(img)
        img = self._blur(img)
        seg_fields = results.get('seg_fields', [])
        segs = [results[key] for key in seg_fields]
        img, segs = self._perspective(img, segs)
        results['img'] = img
        for key, seg in zip(seg_fields, segs):
            results[key] = seg
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(brightness={self.brightness}, "
            f"contrast={self.contrast}, saturation={self.saturation}, "
            f"hue={self.hue}, gamma={self.gamma}, blur_prob={self.blur_prob}, "
            f"noise_std={self.noise_std}, perspective_prob={self.perspective_prob})"
        )
