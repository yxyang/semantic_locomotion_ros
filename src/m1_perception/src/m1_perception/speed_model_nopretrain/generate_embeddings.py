"""Evaluates TF-based FCHarDNet Model"""
import datetime
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
flags.DEFINE_string('pca_data_dir', None, 'path to pca data.')
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
  model.load_pca_data(FLAGS.pca_data_dir)

  image_paths = sorted(recursive_glob(FLAGS.image_dir, suffix='png'))
  timestamps, embeddings = [], []
  files_count = 0

  for image_path in tqdm(image_paths):
    image = Image.open(image_path)
    image = np.array(image).astype(np.uint8)
    image = image.astype(np.float32) / 255.
    image = image[np.newaxis, ...]
    pred_mask = model.get_embedding_lowdim(image).numpy()[0]
    embeddings.append(pred_mask)
    filename = os.path.split(image_path)[1]  # Extract filename
    timestamp = datetime.datetime.strptime(filename[4:-11],
                                           '%Y_%m_%d_%H_%M_%S_%f')
    timestamps.append(timestamp)

    if len(embeddings) >= 1000:
      if len(embeddings) != len(timestamps):
        raise RuntimeError("Embeddings and Original files do not match.")
      embeddings = np.stack(embeddings, axis=0)
      np.savez(os.path.join(FLAGS.output_dir,
                            'embeddings_{}.npz'.format(files_count)),
               embeddings=embeddings,
               timestamps=timestamps)
      embeddings = []
      timestamps = []
      files_count += 1


if __name__ == "__main__":
  app.run(main)
