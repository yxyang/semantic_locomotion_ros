"""Utilities to generate segmentation masks."""
import cv2
import numpy as np

def get_segmentation_mask(height, width):
  """Generates segmentation mask based on image shape."""
  mask = np.zeros((height, width))
  left_bottom = [0.25 * height, height]
  right_bottom = [0.75 * height, height]
  left_top = [
      0.4 * height,
      0.75 * height,
  ]
  right_top = [
      0.6 * height,
      0.75 * height,
  ]
  cv2.fillConvexPoly(
      mask,
      np.array([left_bottom, right_bottom, right_top,
                left_top]).astype(np.int32), 1)
  return mask
