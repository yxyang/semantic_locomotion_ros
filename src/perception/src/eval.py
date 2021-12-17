"""Evaluate trained segmentation model."""
import multiprocessing
import os

import numpy as np
import torch
from absl import app
from absl import flags
from ml_collections import config_flags
from PIL import Image
from torch.utils import data
from tqdm import tqdm

from loader import get_loader
from metrics import RunningScore
from models import get_model
from utils import convert_state_dict

config_flags.DEFINE_config_file("config", "configs/rugd.py",
                                "Experiment config.")
flags.DEFINE_string("model_path", "logs/hardnet_rugd_best_model.pkl",
                    "model to test")
flags.DEFINE_string("eval_dir", "eval", "the data split to evaluate on.")
flags.DEFINE_bool("save_image", True, "whether to save resulting image.")
flags.DEFINE_integer('batch_size', 10,
                     "number of parallel images to evaluate.")
FLAGS = flags.FLAGS

torch.backends.cudnn.benchmark = True


def save_image(args):
  """Saves model outputs to disk."""
  image, annotation_2d, fname, loader = args
  decoded = loader.decode_segmap_id(annotation_2d)
  output_dir = "./out_predID/"
  Image.fromarray(decoded.astype(np.uint8)).save(output_dir + fname)
  decoded = loader.decode_segmap(annotation_2d)
  img_input = image.transpose(1, 2, 0)
  blend = np.concatenate((img_input, decoded), axis=1)
  fname_new = fname
  fname_new = fname_new[:-4] + ".jpg"
  output_dir = "./out_rgb/"
  output_path = os.path.join(output_dir, fname_new)
  Image.fromarray(blend.astype(np.uint8)).save(output_path)


def main(_):
  config = FLAGS.config
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Setup Dataloader
  data_loader = get_loader(config["data"]["dataset"])
  data_path = config["data"]["path"]

  loader = data_loader(data_path,
                       split=FLAGS.eval_dir,
                       is_transform=True,
                       augmentations=None)

  n_classes = loader.n_classes
  valloader = data.DataLoader(loader,
                              batch_size=FLAGS.batch_size,
                              num_workers=1)
  running_metrics = RunningScore(n_classes)

  # Setup Model
  model = get_model(config["model"], n_classes).to(device)
  state = convert_state_dict(torch.load(FLAGS.model_path)["model_state"])
  model.load_state_dict(state)
  model.eval()
  model.to(device)
  total_params = sum(p.numel() for p in model.parameters())
  print('Parameters: ', total_params)
  torch.backends.cudnn.benchmark = True

  # Setup Output
  output_dir = "./out_predID/"
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  if FLAGS.save_image:
    output_dir = "./out_rgb/"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  loaders = [loader] * FLAGS.batch_size
  for images, labels, fnames in tqdm(valloader):
    gpu_images = images.to(device)
    with torch.no_grad():
      outputs = model(gpu_images)
    torch.cuda.synchronize()
    pred = outputs.data.max(1)[1].cpu().numpy()
    gt = labels.numpy()
    running_metrics.update(gt, pred)
    if FLAGS.save_image:
      outputs = outputs.data.max(1)[1].cpu().numpy()
      p = multiprocessing.Pool(FLAGS.batch_size)
      p.map(save_image, zip(images.numpy(), outputs, fnames, loaders))

  score, class_iou = running_metrics.get_scores()

  for k, v in score.items():
    print(k, v)

  for i in range(n_classes):
    print(i, class_iou[i])


if __name__ == "__main__":
  app.run(main)
