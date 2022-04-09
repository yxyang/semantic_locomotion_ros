"""Data augmentations to prevent overfitting."""
import numbers
import random

import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms.functional as tf


class RandomCrop:
  """Randomly crops the input image."""
  def __init__(self, size=(500, 600), padding=0):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size
    self.padding = padding

  def __call__(self, img, mask):
    if self.padding > 0:
      img = ImageOps.expand(img, border=self.padding, fill=0)
      mask = ImageOps.expand(mask, border=self.padding, fill=0)

    assert img.size == mask.size
    w, h = img.size
    ch, cw = self.size
    if w == cw and h == ch:
      return img, mask
    if w < cw or h < ch:
      pw = cw - w if cw > w else 0
      ph = ch - h if ch > h else 0
      padding = (pw, ph, pw, ph)
      # pytype: disable=wrong-arg-types
      img = ImageOps.expand(img, padding, fill=0)
      mask = ImageOps.expand(mask, padding, fill=250)
      # pytype: enable=wrong-arg-types
      w, h = img.size
      assert img.size == mask.size

    x1 = random.randint(0, w - cw)
    y1 = random.randint(0, h - ch)
    img = img.crop((x1, y1, x1 + cw, y1 + ch))
    mask = mask.crop((x1, y1, x1 + cw, y1 + ch))
    return img.resize((w, h), resample=Image.BILINEAR), mask.resize(
        (w, h), resample=Image.NEAREST)


class GaussianBlur:
  def __init__(self, max_sigma=1, kernel_size=11):
    self.kernel_size = kernel_size
    self.max_sigma = max_sigma

  def __call__(self, img, mask):
    img = tf.gaussian_blur(img,
                           kernel_size=self.kernel_size,
                           sigma=np.random.uniform(0, self.max_sigma))
    return img, mask


class AdjustGamma:
  """Adjusts the gamma value of image."""
  def __init__(self, gamma=1.1):
    self.gamma = gamma

  def __call__(self, img, mask):
    assert img.size == mask.size
    return (tf.adjust_gamma(img, random.uniform(1 / self.gamma,
                                                self.gamma)), mask)


class AdjustSaturation:
  """Adjusts the saturation of image."""
  def __init__(self, saturation=0.1):
    self.saturation = saturation

  def __call__(self, img, mask):
    assert img.size == mask.size
    return (
        tf.adjust_saturation(
            img, random.uniform(1 - self.saturation, 1 + self.saturation)),
        mask,
    )


class AdjustHue:
  """Adjusts the hue of image."""
  def __init__(self, hue=0.1):
    self.hue = hue

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness:
  """Adjusts the brightness of image."""
  def __init__(self, bf=0.1):
    self.bf = bf

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_brightness(img, random.uniform(1 - self.bf,
                                                    1 + self.bf)), mask


class AdjustContrast:
  """Adjusts the contrast of image."""
  def __init__(self, cf=0.1):
    self.cf = cf

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_contrast(img, random.uniform(1 - self.cf,
                                                  1 + self.cf)), mask


class RandomHorizontallyFlip:
  """Flips the image horizontally with probability p."""
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, img, mask):
    if random.random() < self.p:
      return (img.transpose(Image.FLIP_LEFT_RIGHT),
              mask.transpose(Image.FLIP_LEFT_RIGHT))
    return img, mask


class RandomRotate:
  """Randomly rotate the image."""
  def __init__(self, degree=10):
    self.degree = degree

  def __call__(self, img, mask):
    rotate_degree = random.random() * 2 * self.degree - self.degree
    return (
        tf.rotate(
            img,
            angle=rotate_degree,
            interpolation=Image.BILINEAR,
            expand=False,
        ),
        tf.rotate(
            mask,
            angle=rotate_degree,
            interpolation=Image.NEAREST,
            expand=False,
        ),
    )
