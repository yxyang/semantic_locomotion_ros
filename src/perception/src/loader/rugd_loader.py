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
      img_size=(1024, 2048),
      augmentations=None,
      img_norm=False,
      version="cityscapes",
      test_mode=False,
  ):
    """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
    self.root = root
    self.split = split
    self.is_transform = is_transform
    self.augmentations = augmentations
    self.img_norm = img_norm
    self.n_classes = 25
    self.img_size = img_size if isinstance(img_size, tuple) else (img_size,
                                                                  img_size)
    self.mean = np.array(self.mean_rgb[version])
    self.files = {}

    self.images_base = os.path.join(self.root, "frames", self.split)
    self.annotations_base = os.path.join(self.root, "annotations_2d",
                                         self.split)

    self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

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
    img_path = self.files[self.split][index].rstrip()
    lbl_path = img_path.replace('frames', 'annotations_2d')
    name = img_path.split(os.sep)[-1][:-4] + ".png"

    img = Image.open(img_path)
    img = np.array(img, dtype=np.uint8)

    lbl = Image.open(lbl_path)
    lbl = np.array(lbl, dtype=np.uint8)
    lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
    if self.augmentations is not None:
      img, lbl = self.augmentations(img, lbl)

    if self.is_transform:
      img, lbl = self.transform(img, lbl)

    return img, lbl, name

  def transform(self, img, lbl):
    """transform

        :param img:
        :param lbl:
    """
    img = np.array(
        Image.fromarray(img).resize(
            (self.img_size[1], self.img_size[0])))  # uint8 with RGB mode
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)

    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]

    if self.img_norm:
      img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)

    classes = np.unique(lbl)
    lbl = lbl.astype(float)
    lbl = np.array(
        Image.fromarray(lbl).resize((self.img_size[1], self.img_size[0]),
                                    resample=Image.NEAREST))
    lbl = lbl.astype(int)

    if not np.all(classes == np.unique(lbl)):
      print("WARN: resizing labels yielded fewer classes")

    if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
      print(np.unique(lbl[lbl != self.ignore_index]))
      raise ValueError("Segmentation map contained invalid class values")

    img = torch.from_numpy(img).float()
    lbl = torch.from_numpy(lbl).long()

    return img, lbl

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
    imgs, labels, _ = data_samples
    imgs = imgs.numpy()[:, ::-1, :, :]
    imgs = np.transpose(imgs, [0, 2, 3, 1])
    imgs = imgs.astype(np.int)
    labels_decoded = dst.decode_segmap(labels.numpy())
    f, axarr = plt.subplots(bs, 2)
    for j in range(bs):
      axarr[j][0].imshow(imgs[j])
      axarr[j][1].imshow(labels_decoded[j])
    plt.show()
    a = input()
    if a == "ex":
      break
    else:
      plt.close()
