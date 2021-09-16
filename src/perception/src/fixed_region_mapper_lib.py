"""Decides gait based on average traversability score of a fixed region."""
import cv2
import numpy as np
import os
import ros_numpy
import rospkg
from sensor_msgs.msg import Image
import torch
from typing import Tuple
import yaml

from models import get_model
from utils import convert_state_dict


def _convert_segmentation_map(raw_segmentation_map):
  color_val = np.array([[0., 1., 0.], [1., 1., 0.], [1., .5, 0.], [1., 0., 0.],
                        [0., 0., 1.]])
  return color_val[raw_segmentation_map]


class FixedRegionMapper:
  """Computes traversability score based on fixed image crop."""
  def __init__(self, cfg_dir='model_configs/hardnet.yml'):
    self._camera_image = None
    self._image_height = None
    self._image_width = None
    self._image_array = None
    self._segmentation_model = self._load_segmentation_model(cfg_dir)
    self._last_segmentation_map = None

  def _load_segmentation_model(self, cfg_dir: str) -> torch.nn.Module:
    rospack = rospkg.RosPack()
    package_path = os.path.join(rospack.get_path('perception'), "src")

    cfg = yaml.load(open(os.path.join(package_path, cfg_dir), 'r'))
    n_classes = 5
    if torch.cuda.is_available():
      self._device = torch.device("cuda")
    else:
      self._device = torch.device("cpu")

    model = get_model(cfg["model"], n_classes).to(self._device)
    model_dir = os.path.join(package_path, "model_ckpts",
                             "hardnet_rugd4_aug5_sky.pkl")
    state = convert_state_dict(torch.load(model_dir)["model_state"])
    model.load_state_dict(state)
    return model

  def _get_segmentation_mask(self) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((self._image_height, self._image_width))
    mask_boundary = np.zeros_like(mask)

    left_bottom = [0, self._image_height]
    right_bottom = [self._image_width, self._image_height]
    left_top = [
        0.33 * self._image_width,
        0.5 * self._image_height,
    ]
    right_top = [
        0.66 * self._image_width,
        0.5 * self._image_height,
    ]
    cv2.fillConvexPoly(
        mask,
        np.array([left_bottom, right_bottom, right_top,
                  left_top]).astype(np.int32), 1)
    cv2.polylines(mask_boundary, [
        np.array([left_bottom, right_bottom, right_top, left_top]).reshape(
            (-1, 1, 2)).astype(np.int32),
    ],
                  isClosed=True,
                  color=1,
                  thickness=10)

    return mask, mask_boundary

  def _compute_segmentation_map(self) -> np.ndarray:
    img = self._image_array.copy()
    img = np.rollaxis(img, -1, 0)
    img = img[np.newaxis, ...]
    img = torch.tensor(img).to(self._device)
    seg_map = self._segmentation_model(img)
    seg_map = np.array(seg_map.cpu().detach().numpy())
    seg_map = seg_map[0]
    seg_map = np.argmax(seg_map, axis=0)
    return seg_map

  def get_segmentation_result(self) -> Tuple[np.ndarray, float]:
    segmentation_map = self._compute_segmentation_map()
    mask, boundary = self._get_segmentation_mask()
    rgb_segmentation_map = _convert_segmentation_map(segmentation_map)
    traversability_score = np.sum(mask * segmentation_map) / np.sum(mask)

    disp_image = np.concatenate((self._image_array, rgb_segmentation_map),
                                axis=1)
    boundary = np.concatenate((boundary, boundary), axis=1)
    boundary = np.stack((boundary, boundary, boundary), axis=-1)
    visualization = (np.array(disp_image + boundary) * 255).astype(np.uint8)
    return traversability_score, ros_numpy.msgify(Image,
                                                  visualization,
                                                  encoding="rgb8")

  @property
  def camera_image(self):
    return self._camera_image

  @property
  def image_array(self):
    return self._image_array

  def set_camera_image(self, image):
    self._camera_image = image
    self._image_height = image.height
    self._image_width = image.width
    self._image_array = np.frombuffer(image.data, dtype=np.uint8).reshape(
        image.height, image.width, -1).astype(np.float32) / 255.
