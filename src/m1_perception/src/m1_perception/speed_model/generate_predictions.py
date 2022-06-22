"""Generates speed predictions from camera."""
import os

from absl import app
from absl import flags

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

from m1_perception.fchardnet.data_loader.data_utils import recursive_glob
from m1_perception.fchardnet.model import HardNet
from m1_perception.speed_model.model import SpeedModel
from m1_perception.speed_model import mask_utils

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

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


def compute_gait(desired_vel):
  desired_vel = np.clip(desired_vel, 0.5, 1.5)
  stepping_freq = 2.8 + 0.7 * (desired_vel - 0.5)
  swing_height = 0.16 - 0.04 * (desired_vel - 0.5)
  base_height = 0.29 - 0.03 * (desired_vel - 0.5)

  return '           SF:{:.2f}, SH:{:.2f}, BH:{:.2f}'.format(
      stepping_freq, swing_height, base_height)


def generate_plot_and_save(output_path, image, heatmap, pred_speed):
  """Generates speed plot, heatmap and original image side-by-side."""
  if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
  # fig = plt.figure(figsize=(8 * 1.5, 2.25 * 1.5))

  fig, axs = plt.subplots(
      3,
      1,
      constrained_layout=True,
      figsize=(4, 8),
      gridspec_kw={
          'height_ratios': [3, 3, 0.5],
          'hspace': 0.1,
          # 'wspace': 0.1
      })
  # plt.subplot(1, 2, 1)
  axs[0].axis('off')
  axs[0].imshow(image)

  axs[0].set_title("Camera Image", fontsize=16)

  # plt.subplot(1, 2, 2)
  colorbar = axs[1].imshow(heatmap, vmin=0.5, vmax=1.5)
  axs[1].axis('off')
  cbplot = plt.colorbar(colorbar,
                        location='bottom',
                        ax=axs[1],
                        aspect=12,
                        orientation='horizontal',
                        pad=0.1)
  cbplot.ax.plot([pred_speed, pred_speed], [-0.5, 2.5], 'r', linewidth=3)
  cbplot.ax.set_xlabel('Forward Speed / (m/s)', labelpad=-60, fontsize=16)

  axs[1].set_title("Speed Map", fontsize=16)

  axs[2].axis('off')
  axs[2].set_title("Desired Gait", fontsize=16)
  # axs[2].text(40, 20, "Desired Gait")
  axs[2].text(0,
              0,
              compute_gait(pred_speed),
              fontsize=14,
              horizontalalignment='left')

  # plt.subplot(1, 3, 3)
  # plt.errorbar(['Speed Command'],
  #              pred_speed,
  #              yerr=0,
  #              capsize=10,
  #              capthick=5,
  #              elinewidth=5,
  #              markersize=10,
  #              fmt='o')
  # plt.xticks(fontsize=20)
  # plt.ylim(0, 2)
  # plt.ylabel('Predicted Speed (m/s)', fontsize=15)

  plt.savefig(output_path, format='png', dpi=160)
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
