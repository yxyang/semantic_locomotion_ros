"""Utility for loading cityscape data."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from perception.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
from perception.utils import recursive_glob


class CityscapesLoader(data.Dataset):
  """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

  colors = [  # [  0,   0,   0],
      [128, 64, 128],
      [244, 35, 232],
      [70, 70, 70],
      [102, 102, 156],
      [190, 153, 153],
      [153, 153, 153],
      [250, 170, 30],
      [220, 220, 0],
      [107, 142, 35],
      [152, 251, 152],
      [0, 130, 180],
      [220, 20, 60],
      [255, 0, 0],
      [0, 0, 142],
      [0, 0, 70],
      [0, 60, 100],
      [0, 80, 100],
      [0, 0, 230],
      [119, 11, 32],
  ]

  label_colours = dict(zip(range(19), colors))

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
      image_norm=True,
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
    self.n_classes = 19
    self.image_size = image_size if isinstance(
        image_size, tuple) else (image_size, image_size)
    self.mean = np.array(self.mean_rgb[version])
    self.files = {}

    self.images_base = os.path.join(self.root, "leftimage8bit", self.split)
    self.annotations_base = os.path.join(self.root, "gtFine", self.split)

    self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

    self.void_classes = [
        0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1
    ]
    self.valid_classes = [
        7,
        8,
        11,
        12,
        13,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        31,
        32,
        33,
    ]

    #self.void_classes = [ 255]
    #self.valid_classes = [i for i in range(19)]
    self.class_names = [
        "unlabelled",
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    self.ignore_index = 250
    self.class_map = dict(zip(self.valid_classes, range(19)))

    if not self.files[split]:
      raise Exception("No files for split=[%s] found in %s" %
                      (split, self.images_base))

    print("Found %d %s images" % (len(self.files[split]), split))

  def __len__(self):
    """__len__"""
    return len(self.files[self.split])

  def __getitem__(self, index):
    """__getitem__

        :param index:
        """
    image_path = self.files[self.split][index].rstrip()
    label_path = os.path.join(
        self.annotations_base,
        image_path.split(os.sep)[-2],
        os.path.basename(image_path)[:-15] + "gtFine_labelIds.png",
    )
    label_name = image_path.split(os.sep)[-1][:-4] + ".png"

    image = Image.open(image_path)
    image = np.array(image, dtype=np.uint8)

    label = Image.open(label_path)
    label = self.encode_segmap(np.array(label, dtype=np.uint8))

    if self.augmentations is not None:
      image, label = self.augmentations(image, label)

    if self.is_transform:
      image, label = self.transform(image, label)

    return image, label, label_name

  def transform(self, image, label):
    """transform

        :param image:
        :param label:
        """
    image = np.array(
        Image.fromarray(image).resize(
            (self.image_size[1], self.image_size[0])))  # uint8 with RGB mode
    image = image[:, :, ::-1]  # RGB -> BGR
    image = image.astype(np.float64)

    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]

    if self.image_norm:
      image = (image - mean) / std

    # NHWC -> NCHW
    image = image.transpose(2, 0, 1)

    classes = np.unique(label)
    label = label.astype(float)
    label = np.array(
        Image.fromarray(label).resize((self.image_size[1], self.image_size[0]),
                                      resample=Image.NEAREST))
    label = label.astype(int)

    if not np.all(classes == np.unique(label)):
      print("WARN: resizing labels yielded fewer classes")

    if not np.all(
        np.unique(label[label != self.ignore_index]) < self.n_classes):
      print("after det", classes, np.unique(label))
      raise ValueError("Segmentation map contained invalid class values")

    image = torch.from_numpy(image).float()
    label = torch.from_numpy(label).long()

    return image, label

  def decode_segmap(self, temp):
    """Decodes segmentation class into rgb segmentation map."""
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, self.n_classes):
      r[temp == l] = self.label_colours[l][0]
      g[temp == l] = self.label_colours[l][1]
      b[temp == l] = self.label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
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
    for valic_class in self.valid_classes:
      mask[mask == valic_class] = self.class_map[valic_class]
    return mask


if __name__ == "__main__":
  default_augmentations = Compose(
      [Scale(2048), RandomRotate(10),
       RandomHorizontallyFlip(0.5)])

  local_path = "/home/yxyang/Downloads/gtFine_trainvaltest"
  dst = CityscapesLoader(local_path,
                         is_transform=True,
                         augmentations=default_augmentations)
  bs = 4
  trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
  for i, data_samples in enumerate(trainloader):
    images, labels, name = data_samples
    images = images.numpy()[:, ::-1, :, :]
    images = np.transpose(images, [0, 2, 3, 1])
    f, axarr = plt.subplots(bs, 2)
    for j in range(bs):
      axarr[j][0].imshow(images[j])
      axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
    plt.show()
    a = input()
    if a == "ex":
      break
    else:
      plt.close()
