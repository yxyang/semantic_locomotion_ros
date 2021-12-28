"""Utility for loading cityscape data."""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils import data

from augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
from utils import recursive_glob


class RUGDLoader(data.Dataset):
  """Loads data from RUGD dataset for processing."""
  mean_rgb = {
      "pascal": [103.939, 116.779, 123.68],
      "cityscapes": [0.0, 0.0, 0.0],
  }  # pascal mean for PSPNet and ICNet pre-trained model

  def __init__(
      self,
      root,
      split="train",
      is_transform=False,
      image_size=(1024, 2048),
      augmentations=None,
      image_norm=False,
      version="cityscapes",
      test_mode=False,
  ):
    """__init__

        :param root:
        :param split:
        :param is_transform:
        :param image_size:
        :param augmentations
        """
    self.root = root
    self.split = split
    self.is_transform = is_transform
    self.augmentations = augmentations
    self.image_norm = image_norm
    self.n_classes = 25
    self.image_size = image_size if isinstance(
        image_size, tuple) else (image_size, image_size)
    self.mean = np.array(self.mean_rgb[version])
    self.files = {}

    self.images_base = os.path.join(self.root, "frames", self.split)
    self.annotations_base = os.path.join(self.root, "annotations_2d",
                                         self.split)

    self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
    if split == "val":
      self.files[split] = sorted(self.files[split])

    self._load_segmentation_classes(os.path.join(self.root, "annotations"))
    self.ignore_index = 250

    if not self.files[split]:
      raise Exception("No files for split=[%s] found in %s" %
                      (split, self.images_base))

    print("Found %d %s images" % (len(self.files[split]), split))

  def _load_segmentation_classes(self, annotations_base):
    self.valid_classes, self.class_names, self.class_colors = [], [], []
    with open(os.path.join(annotations_base,
                           'RUGD_annotation-colormap.txt')) as csvfile:
      reader = csv.reader(csvfile, delimiter=' ')
      for row in reader:
        self.valid_classes.append(int(row[0]))
        self.class_names.append(row[1])
        self.class_colors.append([int(row[2]), int(row[3]), int(row[4])])
    self.void_classes = []

  def __len__(self):
    """__len__"""
    return len(self.files[self.split])

  def __getitem__(self, index):
    """__getitem__

        :param index:
        """
    image_path = self.files[self.split][index].rstrip()
    lbl_path = image_path.replace('frames', 'annotations_2d')
    name = image_path.split(os.sep)[-1][:-4] + ".png"

    image = Image.open(image_path)
    image = np.array(image, dtype=np.uint8)

    if os.path.exists(lbl_path):
      lbl = Image.open(lbl_path)
      lbl = np.array(lbl, dtype=np.uint8)
      lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
    else:
      lbl = np.zeros((image.shape[:2]))  #pylint: disable=unsubscriptable-object
    if self.augmentations is not None:
      image, lbl = self.augmentations(image, lbl)

    if self.is_transform:
      image, lbl = self.transform(image, lbl)

    return image, lbl, name

  def transform(self, image, lbl):
    """transform

        :param image:
        :param lbl:
    """
    image = np.array(
        Image.fromarray(image).resize(
            (self.image_size[1], self.image_size[0])))  # uint8 with RGB mode
    image = image.astype(np.float64)

    # Debug: auto brightness
    img_float = image / 255.
    brightness = np.mean(0.2126 * img_float[..., 2] +
                         0.7152 * img_float[..., 1] +
                         0.0722 * img_float[..., 0])
    desired_brightness = 0.5
    img_float = np.clip(img_float * desired_brightness / brightness, 0, 1)
    image = img_float * 255

    # NHWC -> NCHW
    image = image.transpose(2, 0, 1)

    classes = np.unique(lbl)
    lbl = lbl.astype(float)
    lbl = np.array(
        Image.fromarray(lbl).resize((self.image_size[1], self.image_size[0]),
                                    resample=Image.NEAREST))
    lbl = lbl.astype(int)

    if not np.all(classes == np.unique(lbl)):
      print("WARN: resizing labels yielded fewer classes")

    if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
      print(np.unique(lbl[lbl != self.ignore_index]))
      raise ValueError("Segmentation map contained invalid class values")

    image = torch.from_numpy(image).float()
    lbl = torch.from_numpy(lbl).long()

    return image, lbl

  def decode_segmap(self, segmap):
    """Decodes segmentation map (hxw) to rgb image (hxwx3)"""
    r = segmap.copy()
    g = segmap.copy()
    b = segmap.copy()
    for l in range(0, self.n_classes):
      r[segmap == l] = self.class_colors[l][0]
      g[segmap == l] = self.class_colors[l][1]
      b[segmap == l] = self.class_colors[l][2]
    rgb = np.zeros(np.concatenate([segmap.shape, [3]]))
    rgb[..., 0] = r / 255.0
    rgb[..., 1] = g / 255.0
    rgb[..., 2] = b / 255.0
    return rgb

  def decode_segmap_id(self, temp):
    ids = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)
    for l in range(0, self.n_classes):
      ids[temp == l] = self.valid_classes[l]
    return ids

  def encode_segmap(self, mask):
    # Put all void classes to zero
    for void_class in self.void_classes:
      mask[mask == void_class] = self.ignore_index
    return mask


if __name__ == "__main__":
  default_augmentations = Compose([
      Scale(2048),
      RandomRotate(10),
      RandomHorizontallyFlip(0.5),
  ])

  local_path = "data/RUGD"
  dst = RUGDLoader(local_path,
                   is_transform=True,
                   augmentations=default_augmentations)
  bs = 4
  trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
  for i, data_samples in enumerate(trainloader):
    images, labels, _ = data_samples
    images = images.numpy()[:, ::-1, :, :]
    images = np.transpose(images, [0, 2, 3, 1])
    images = images.astype(np.int)
    labels_decoded = dst.decode_segmap(labels.numpy())
    f, axarr = plt.subplots(bs, 2)
    for j in range(bs):
      axarr[j][0].imshow(images[j])
      axarr[j][1].imshow(labels_decoded[j])
    plt.show()
    a = input()
    if a == "ex":
      break
    else:
      plt.close()
