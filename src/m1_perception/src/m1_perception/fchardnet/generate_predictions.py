"""Evaluates TF-based FCHarDNet Model"""
import os

from absl import app
from absl import flags

import numpy as np
from PIL import Image
from tqdm import tqdm

from m1_perception.fchardnet.data_loader.data_utils import recursive_glob
from m1_perception.fchardnet.model import HardNet

flags.DEFINE_string('model_dir', None, 'path to saved model.')
flags.DEFINE_string('image_dir', None, 'image directory.')
flags.DEFINE_string('output_dir', None, 'output directory')
FLAGS = flags.FLAGS

CLASS_COLORS = np.array([[0, 0, 0], [108, 64, 20],
                         [255, 229, 204], [0, 102, 0], [0, 255, 0],
                         [0, 153, 153], [0, 128, 255], [0, 0, 255],
                         [255, 255, 0], [255, 0, 127], [64, 64, 64],
                         [255, 128, 0], [255, 0, 0], [153, 76,
                                                      0], [102, 102, 0],
                         [102, 0, 0], [0, 255, 128], [204, 153, 255],
                         [102, 0, 204], [255, 153, 204], [0, 102, 102],
                         [153, 204, 255], [102, 255, 255], [101, 101, 11],
                         [114, 85, 47]])


def decode_segmap(pred_mask):
  """Decode HxWxclass mask to RGB segmentation map."""
  r = np.zeros_like(pred_mask)
  g = np.zeros_like(pred_mask)
  b = np.zeros_like(pred_mask)
  for class_id, class_color in enumerate(CLASS_COLORS):
    r[pred_mask == class_id] = class_color[0]
    g[pred_mask == class_id] = class_color[1]
    b[pred_mask == class_id] = class_color[2]
  return np.stack((r, g, b), axis=-1).astype(np.uint8)


def main(argv):
  del argv  # unused

  model = HardNet()
  model.load_weights(FLAGS.model_dir)

  image_paths = sorted(recursive_glob(FLAGS.image_dir, suffix='png'))

  for image_path in tqdm(image_paths):
    image = Image.open(image_path)
    input_image = np.array(image).astype(np.uint8)
    image = input_image.astype(np.float32) / 255.
    image = image[np.newaxis, ...]
    pred_mask = model(image).numpy()[0]
    pred_mask = np.argmax(pred_mask, axis=-1)

    pred_mask_rgb = decode_segmap(pred_mask)
    output_path = image_path.replace(FLAGS.image_dir, FLAGS.output_dir)

    if not os.path.exists(os.path.dirname(output_path)):
      os.makedirs(os.path.dirname(output_path))
    output_image = np.concatenate((input_image, pred_mask_rgb),
                                  axis=1)
    Image.fromarray(output_image).save(output_path)


if __name__ == "__main__":
  app.run(main)
