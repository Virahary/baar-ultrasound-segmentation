"""Paired geometric and image-intensity augmentation for training."""
import cv2
import numpy as np
def gaussian_filter_cv(img, sigma):
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

class MedicalImageAugmentation:

    def __init__(self, rotation_range=15, scale_range=(0.9, 1.1), translate_range=0.1, shear_range=5, elastic_alpha=50, elastic_sigma=5, gamma_range=(0.8, 1.2), brightness_range=0.1, contrast_range=(0.9, 1.1), gaussian_noise_std=0.02, p_affine=0.5, p_elastic=0.3, p_gamma=0.5, p_brightness=0.3, p_contrast=0.3, p_noise=0.2, p_flip_h=0.5, p_flip_v=0.0):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.shear_range = shear_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gaussian_noise_std = gaussian_noise_std
        self.p_affine = p_affine
        self.p_elastic = p_elastic
        self.p_gamma = p_gamma
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_noise = p_noise
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v

    def __call__(self, image, mask):
        if np.random.rand() < self.p_flip_h:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < self.p_flip_v:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        if np.random.rand() < self.p_affine:
            image, mask = self._affine_transform(image, mask)
        if np.random.rand() < self.p_elastic:
            image, mask = self._elastic_deformation(image, mask)
        if np.random.rand() < self.p_gamma:
            image = self._gamma_correction(image)
        if np.random.rand() < self.p_brightness:
            image = self._brightness_adjust(image)
        if np.random.rand() < self.p_contrast:
            image = self._contrast_adjust(image)
        if np.random.rand() < self.p_noise:
            image = self._add_gaussian_noise(image)
        image = np.clip(image, 0, 1)
        mask = (mask > 0.5).astype(np.uint8)
        return (image, mask)

    def _affine_transform(self, image, mask):
        h, w = image.shape
        center = (w / 2, h / 2)
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        scale = np.random.uniform(*self.scale_range)
        tx = np.random.uniform(-self.translate_range, self.translate_range) * w
        ty = np.random.uniform(-self.translate_range, self.translate_range) * h
        shear = np.random.uniform(-self.shear_range, self.shear_range) * np.pi / 180
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        shear_matrix = np.array([[1, np.tan(shear), 0], [0, 1, 0]], dtype=np.float32)
        M = np.vstack([M, [0, 0, 1]])
        shear_matrix = np.vstack([shear_matrix, [0, 0, 1]])
        M = (shear_matrix @ M)[:2]
        image_aug = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_aug = cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (image_aug, mask_aug.astype(np.uint8))

    def _elastic_deformation(self, image, mask):
        h, w = image.shape
        dx = gaussian_filter_cv(np.random.rand(h, w).astype(np.float32) * 2 - 1, self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter_cv(np.random.rand(h, w).astype(np.float32) * 2 - 1, self.elastic_sigma) * self.elastic_alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        image_aug = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_aug = cv2.remap(mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (image_aug.astype(np.float32), (mask_aug > 0.5).astype(np.uint8))

    def _gamma_correction(self, image):
        gamma = np.random.uniform(*self.gamma_range)
        return np.power(image + 1e-08, gamma)

    def _brightness_adjust(self, image):
        delta = np.random.uniform(-self.brightness_range, self.brightness_range)
        return image + delta

    def _contrast_adjust(self, image):
        factor = np.random.uniform(*self.contrast_range)
        mean = image.mean()
        return (image - mean) * factor + mean

    def _add_gaussian_noise(self, image):
        noise = np.random.normal(0, self.gaussian_noise_std, image.shape)
        return image + noise
