"""Generates speed predictions from camera."""
import os

from absl import app
from absl import flags

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from m1_perception.fchardnet.data_loader.data_utils import recursive_glob
from m1_perception.fchardnet.model import HardNet
from m1_perception.speed_model_class_category.model import SpeedModel
from m1_perception.speed_model_class_category import mask_utils

flags.DEFINE_string('vision_model_dir', 'checkpoints/vision_model/cp-99.ckpt',
                    'path to vision model.')
flags.DEFINE_string('speed_model_dir',
                    'checkpoints/speed_model_summer_cleaned/cp-59.ckpt',
                    'path to speed model.')
flags.DEFINE_string(
    'image_dir', '/home/yxyang/research/semantic_locomotion_ros/data/'
    'ghost_memory_taylor_demo2/extracted_images', 'path to images.')
flags.DEFINE_string(
    'output_dir', '/home/yxyang/research/semantic_locomotion_ros/data/'
    'ghost_memory_taylor_demo2/predictions', 'output paths.')
FLAGS = flags.FLAGS


def generate_plot_and_save(output_path, image, heatmap, pred_speed):
  """Generates speed plot, heatmap and original image side-by-side."""
  if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
  fig = plt.figure(figsize=(12 * 1.5, 2.25 * 1.5))
  plt.subplot(1, 3, 1)
  plt.axis('off')
  plt.imshow(image)

  plt.subplot(1, 3, 2)
  colorbar = plt.imshow(heatmap, vmin=0.5, vmax=1.5)
  plt.axis('off')
  plt.colorbar(colorbar, location='right')

  plt.subplot(1, 3, 3)
  plt.errorbar(['Speed Command'],
               pred_speed,
               yerr=0,
               capsize=10,
               capthick=5,
               elinewidth=5,
               markersize=10,
               fmt='o')
  plt.xticks(fontsize=20)
  plt.ylim(0, 2)
  plt.ylabel('Predicted Speed (m/s)', fontsize=15)

  plt.savefig(output_path, format='png')
  plt.close(fig)
  del fig


def main(argv):
  matplotlib.use('Agg')
  plt.ioff()
  del argv  # unused

  vision_model = HardNet()
  vision_model.load_weights(FLAGS.vision_model_dir)

  speed_model = SpeedModel(num_hidden_layers=1, dim_hidden=20)
  speed_model.load_weights(FLAGS.speed_model_dir)

  mask = None
  image_paths = sorted(recursive_glob(FLAGS.image_dir, suffix='png'))
  for image_path in tqdm(image_paths):
    image = Image.open(image_path)
    image = np.array(image).astype(np.float32) / 255.
    image = image[np.newaxis, ...]
    if mask is None:
      mask = mask_utils.get_segmentation_mask(height=image.shape[1],
                                              width=image.shape[2])
    pred_embedding = vision_model.get_embedding(image)[0]
    pred_speed_per_pixel = speed_model(pred_embedding).numpy()
    pred_speed = np.sum(pred_speed_per_pixel * mask) / np.sum(mask)

    output_dir = image_path.replace(FLAGS.image_dir, FLAGS.output_dir)
    generate_plot_and_save(output_dir, image[0], pred_speed_per_pixel,
                           pred_speed)


if __name__ == "__main__":
  app.run(main)
