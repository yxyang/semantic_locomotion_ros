#!/usr/bin/env python
"""Matches labeles inferred from prioperception with vision embeddings."""
import datetime
import os

from absl import app
from absl import flags

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from perception.configs import rugd_a1
from perception.models import get_model
from perception.utils import convert_state_dict
from perception.fixed_region_mapper_lib import FixedRegionMapper

flags.DEFINE_string('logdir', None, 'directory for stored images.')
flags.DEFINE_string('output_dir', None, 'where to save output data.')
FLAGS = flags.FLAGS


def load_image(image_path):
  image = Image.open(image_path)
  image = np.array(image, dtype=np.uint8)
  img_float = image / 255.
  brightness = np.mean(0.2126 * img_float[..., 0] +
                       0.7152 * img_float[..., 1] + 0.0722 * img_float[..., 2])
  desired_brightness = 0.66
  img_float = np.clip(img_float * desired_brightness / brightness, 0, 1)
  image = img_float * 255
  return image


def main(argv):
  del argv  # unused

  # Load perception model
  n_classes = 25
  config = rugd_a1.get_config()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = get_model(config["model"], n_classes).to(device)
  state = convert_state_dict(
      torch.load('src/perception/src/perception/saved_models/RUGD_raw_label/'
                 'hardnet_rugd_best_model.pkl')["model_state"])
  # state = convert_state_dict(
  #     torch.load(
  #         'src/perception/src/perception/saved_models/RUGD_a1_sgd_momentum/'
  #         'hardnet_rugd_a1_1500.pkl'
  #     )["model_state"])
  model.load_state_dict(state)
  model.eval()
  model.to(device)
  torch.backends.cudnn.benchmark = True

  # Mapper
  mapper = FixedRegionMapper(config)
  seg_map, _ = mapper.segmentation_mask

  timestamps, embeddings = [], []
  for filename in tqdm(sorted(os.listdir(FLAGS.logdir))):
    timestamp = datetime.datetime.strptime(filename[4:-11],
                                           '%Y_%m_%d_%H_%M_%S_%f')
    timestamps.append(timestamp)

    image = load_image(os.path.join(FLAGS.logdir, filename))
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, ...]
    image = torch.tensor(image).float().to(device)
    embedding = model.get_embedding(image)
    embedding = embedding.cpu().detach().numpy()[0]
    avg_emb = np.sum(embedding * seg_map, axis=((1, 2))) / np.sum(seg_map)
    embeddings.append(avg_emb)

  np.savez(os.path.join(FLAGS.output_dir, 'vision_embeddings.npz'),
           timestamps=np.array(timestamps),
           embeddings=np.array(embeddings))


if __name__ == "__main__":
  app.run(main)
