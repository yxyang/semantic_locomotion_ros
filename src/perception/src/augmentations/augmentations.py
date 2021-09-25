"""Different ways to perform data augmentations."""
import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
  """Composes a list of augmentations."""
  def __init__(self, augmentations):
    self.augmentations = augmentations
    self.PIL2Numpy = False

  def __call__(self, img, mask):
    if isinstance(img, np.ndarray):
      img = Image.fromarray(img, mode="RGB")
      mask = Image.fromarray(mask, mode="L")
      self.PIL2Numpy = True

    assert img.size == mask.size
    for a in self.augmentations:
      img, mask = a(img, mask)

    if self.PIL2Numpy:
      img, mask = np.array(img), np.array(mask, dtype=np.uint8)

    return img, mask


class RandomCrop(object):
  """Randomly crops the input image."""
  def __init__(self, size, padding=0):
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
    return (img.crop(
        (x1, y1, x1 + cw, y1 + ch)), mask.crop((x1, y1, x1 + cw, y1 + ch)))


class AdjustGamma(object):
  """Adjusts the gamma value of image."""
  def __init__(self, gamma):
    self.gamma = gamma

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
  """Adjusts the saturation of image."""
  def __init__(self, saturation):
    self.saturation = saturation

  def __call__(self, img, mask):
    assert img.size == mask.size
    return (
        tf.adjust_saturation(
            img, random.uniform(1 - self.saturation, 1 + self.saturation)),
        mask,
    )


class AdjustHue(object):
  """Adjusts the hue of image."""
  def __init__(self, hue):
    self.hue = hue

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness(object):
  """Adjusts the brightness of image."""
  def __init__(self, bf):
    self.bf = bf

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_brightness(img, random.uniform(1 - self.bf,
                                                    1 + self.bf)), mask


class AdjustContrast(object):
  """Adjusts the contrast of image."""
  def __init__(self, cf):
    self.cf = cf

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_contrast(img, random.uniform(1 - self.cf,
                                                  1 + self.cf)), mask


class CenterCrop(object):
  """Crops the image in the center."""
  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img, mask):
    assert img.size == mask.size
    w, h = img.size
    th, tw = self.size
    x1 = int(round((w - tw) / 2.0))
    y1 = int(round((h - th) / 2.0))
    return (img.crop(
        (x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class RandomHorizontallyFlip(object):
  """Flips the image horizontally with probability p."""
  def __init__(self, p):
    self.p = p

  def __call__(self, img, mask):
    if random.random() < self.p:
      return (img.transpose(Image.FLIP_LEFT_RIGHT),
              mask.transpose(Image.FLIP_LEFT_RIGHT))
    return img, mask


class RandomVerticallyFlip(object):
  """Flips the image vertically with probability p."""
  def __init__(self, p):
    self.p = p

  def __call__(self, img, mask):
    if random.random() < self.p:
      return (img.transpose(Image.FLIP_TOP_BOTTOM),
              mask.transpose(Image.FLIP_TOP_BOTTOM))
    return img, mask


class FreeScale(object):
  """Resizes the image."""
  def __init__(self, size):
    self.size = tuple(reversed(size))  # size: (h, w)

  def __call__(self, img, mask):
    assert img.size == mask.size
    return (img.resize(self.size,
                       Image.BILINEAR), mask.resize(self.size, Image.NEAREST))


class RandomScaleCrop(object):
  """Randomly scale-crop the image."""
  def __init__(self, size):
    self.size = size
    self.crop = RandomCrop(self.size)

  def __call__(self, img, mask):
    assert img.size == mask.size
    r = random.uniform(0.5, 2.0)
    w, h = img.size
    new_size = (int(w * r), int(h * r))
    return self.crop(img.resize(new_size, Image.BILINEAR),
                     mask.resize(new_size, Image.NEAREST))


class RandomTranslate(object):
  """Randomly translate / crop the image."""
  def __init__(self, offset):
    # tuple (delta_x, delta_y)
    self.offset = offset

  def __call__(self, img, mask):
    assert img.size == mask.size
    x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
    y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

    x_crop_offset = x_offset
    y_crop_offset = y_offset
    if x_offset < 0:
      x_crop_offset = 0
    if y_offset < 0:
      y_crop_offset = 0

    cropped_img = tf.crop(
        img,
        y_crop_offset,
        x_crop_offset,
        img.size[1] - abs(y_offset),
        img.size[0] - abs(x_offset),
    )

    padding_tuple = (
        max(-x_offset, 0),
        max(-y_offset, 0),
        max(x_offset, 0),
        max(y_offset, 0),
    )

    return (
        tf.pad(cropped_img, padding_tuple, padding_mode="reflect"),
        tf.affine(
            mask,
            translate=(-x_offset, -y_offset),
            scale=1.0,
            angle=0.0,
            shear=0.0,
            fillcolor=250,
        ),
    )


class RandomRotate(object):
  """Randomly rotate the image."""
  def __init__(self, degree):
    self.degree = degree

  def __call__(self, img, mask):
    rotate_degree = random.random() * 2 * self.degree - self.degree
    return (
        tf.affine(
            img,
            translate=(0, 0),
            scale=1.0,
            angle=rotate_degree,
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
            shear=0.0,
        ),
        tf.affine(
            mask,
            translate=(0, 0),
            scale=1.0,
            angle=rotate_degree,
            resample=Image.NEAREST,
            fillcolor=250,
            shear=0.0,
        ),
    )


class Scale(object):
  """Scale the image."""
  def __init__(self, size):
    self.size = size

  def __call__(self, img, mask):
    assert img.size == mask.size
    w, h = img.size
    if (w >= h and w == self.size) or (h >= w and h == self.size):
      return img, mask
    if w > h:
      ow = self.size
      oh = int(self.size * h / w)
      return (img.resize((ow, oh),
                         Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))
    else:
      oh = self.size
      ow = int(self.size * w / h)
      return (img.resize((ow, oh),
                         Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))


class RandomSizedCrop(object):
  """Randomly size-crop the image."""
  def __init__(self, size):
    self.size = size

  def __call__(self, img, mask):
    assert img.size == mask.size
    for _ in range(10):
      area = img.size[0] * img.size[1]
      target_area = random.uniform(0.45, 1.0) * area
      aspect_ratio = random.uniform(0.5, 2)

      w = int(round(math.sqrt(target_area * aspect_ratio)))
      h = int(round(math.sqrt(target_area / aspect_ratio)))

      if random.random() < 0.5:
        w, h = h, w

      if w <= img.size[0] and h <= img.size[1]:
        x1 = random.randint(0, img.size[0] - w)
        y1 = random.randint(0, img.size[1] - h)

        img = img.crop((x1, y1, x1 + w, y1 + h))
        mask = mask.crop((x1, y1, x1 + w, y1 + h))
        assert img.size == (w, h)

        return (
            img.resize((self.size, self.size), Image.BILINEAR),
            mask.resize((self.size, self.size), Image.NEAREST),
        )

    # Fallback
    scale = Scale(self.size)
    crop = CenterCrop(self.size)
    return crop(*scale(img, mask))


class RandomSized(object):
  """Randomly resize the iamge."""
  def __init__(self, size):
    self.size = size
    self.scale = Scale(self.size)
    self.crop = RandomCrop(self.size)

  def __call__(self, img, mask):
    assert img.size == mask.size

    w = int(random.uniform(0.5, 2) * img.size[0])
    h = int(random.uniform(0.5, 2) * img.size[1])

    img, mask = (img.resize(
        (w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST))

    return self.crop(*self.scale(img, mask))