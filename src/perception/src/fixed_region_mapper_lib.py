"""Decides gait based on average traversability score of a fixed region."""
import os
from typing import Tuple

import cv2
import numpy as np
import rospkg
import rospy
import torch
from sensor_msgs.msg import CompressedImage

from configs import rugd_a1
from models import get_model
from utils import convert_state_dict


def _convert_segmentation_map(raw_segmentation_map):
  color_val = np.array([[176., 0, 0], [176, 118, 0], [118, 176, 0],
                        [0, 176, 0]]) / 255.
  return color_val[raw_segmentation_map]


class FixedRegionMapper:
  """Computes traversability score based on fixed image crop."""
  def __init__(self, config=rugd_a1.get_config()):
    self._camera_image = None
    self._image_height = None
    self._image_width = None
    self._image_array = None
    self._segmentation_model = self._load_segmentation_model(config)
    self._last_segmentation_map = None
    self._processed_image_publisher = rospy.Publisher(
        '/perception/processed_camera_image/compressed',
        CompressedImage,
        queue_size=1)

  def _load_segmentation_model(self, config) -> torch.nn.Module:
    """Loads the trained segmentation model."""
    rospack = rospkg.RosPack()
    package_path = os.path.join(rospack.get_path('perception'), "src")

    n_classes = 4
    if torch.cuda.is_available():
      self._device = torch.device("cuda")
    else:
      self._device = torch.device("cpu")

    model = get_model(config["model"], n_classes).to(self._device)
    model_dir = os.path.join(package_path, "saved_models",
                             "RUGD_a1_sgd_momentum",
                             "hardnet_rugd_a1_1500.pkl")
    state = convert_state_dict(torch.load(model_dir)["model_state"])
    model.load_state_dict(state)
    model.eval()
    return model

  def _get_segmentation_mask(self) -> Tuple[np.ndarray, np.ndarray]:
    """Computes segmentation mask, which is a fixed region in the image."""
    mask = np.zeros((self._image_height, self._image_width))
    mask_boundary = np.zeros_like(mask)

    left_bottom = [0.25 * self._image_width, self._image_height]
    right_bottom = [0.75 * self._image_width, self._image_height]
    left_top = [
        0.4 * self._image_width,
        0.5 * self._image_height,
    ]
    right_top = [
        0.6 * self._image_width,
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
    """Preprocesses image and queries model for segmentation result."""
    img = self._image_array.copy()

    # Normalize on brightness
    img_float = img / 255.
    brightness = np.mean(0.2126 * img_float[..., 0] +
                         0.7152 * img_float[..., 1] +
                         0.0722 * img_float[..., 2])
    print("Before: mean per channel: {}".format(np.mean(img_float,
                                                        axis=(0, 1))))
    desired_brightness = 0.66
    img_float = np.clip(img_float * desired_brightness / brightness, 0, 1)
    print("After: mean per channel: {}".format(np.mean(img_float,
                                                       axis=(0, 1))))
    img = img_float * 255

    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "png"
    msg.data = np.array(cv2.imencode(".png", img)[1]).tostring()
    self._processed_image_publisher.publish(msg)

    # Get the right shape
    img = np.rollaxis(img, -1, 0)
    img = img[np.newaxis, ...]
    img = torch.tensor(img).to(self._device)
    seg_map = self._segmentation_model(img)
    seg_map = np.array(seg_map.cpu().detach().numpy())
    seg_map = seg_map[0]
    seg_map = np.argmax(seg_map, axis=0)
    return seg_map

  def get_segmentation_result(self) -> Tuple[np.ndarray, float]:
    """Returns segmentation result (score and visualization)."""
    segmentation_map = self._compute_segmentation_map()
    mask, _ = self._get_segmentation_mask()
    rgb_segmentation_map = _convert_segmentation_map(segmentation_map)
    traversability_score = np.sum(mask * segmentation_map) / np.sum(mask)
    visualization = (np.array(rgb_segmentation_map) * 255).astype(np.uint8)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "png"
    msg.data = np.array(cv2.imencode(".png", visualization)[1]).tostring()
    return traversability_score, msg

  @property
  def camera_image(self):
    return self._camera_image

  @property
  def image_array(self):
    return self._image_array

  def set_camera_image(self, image):
    self._camera_image = image
    buffer = np.fromstring(image.data, np.uint8)
    cv_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    self._image_array = np.array(cv_image, dtype=np.float32)
    self._image_height = self._image_array.shape[0]  # pylint: disable=E1136
    self._image_width = self._image_array.shape[1]  # pylint: disable=E1136
