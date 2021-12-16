"""Evaluate trained segmentation model."""
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
FLAGS = flags.FLAGS

torch.backends.cudnn.benchmark = True


def main(_):
  config = FLAGS.config
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Setup Dataloader
  data_loader = get_loader(config["data"]["dataset"])
  data_path = config["data"]["path"]

  loader = data_loader(
      data_path,
      split=FLAGS.eval_dir,
      is_transform=True,
  )

  n_classes = loader.n_classes

  valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
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

  for i, (images, labels, fname) in tqdm(enumerate(valloader)):

    images = images.to(device)

    if i == 0:
      with torch.no_grad():
        outputs = model(images)

    torch.cuda.synchronize()

    with torch.no_grad():
      outputs = model(images)

    torch.cuda.synchronize()

    if FLAGS.save_image:
      pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
      decoded = loader.decode_segmap_id(pred)
      output_dir = "./out_predID/"
      if not os.path.exists(output_dir):
        os.mkdir(output_dir)
      Image.fromarray(decoded.astype(np.uint8)).save(output_dir + fname[0])

      decoded = loader.decode_segmap(pred) * 255.
      img_input = np.squeeze(images.cpu().numpy(), axis=0)
      img_input = img_input.transpose(1, 2, 0)
      blend = np.concatenate((img_input, decoded), axis=1)
      fname_new = fname[0]
      fname_new = fname_new[:-4]
      fname_new += '.jpg'
      output_dir = "./out_rgb/"
      output_path = os.path.join(output_dir, fname_new)
      if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
      Image.fromarray(blend.astype(np.uint8)).save(output_path)

    pred = outputs.data.max(1)[1].cpu().numpy()

    gt = labels.numpy()
    running_metrics.update(gt, pred)

  score, class_iou = running_metrics.get_scores()

  for k, v in score.items():
    print(k, v)

  for i in range(n_classes):
    print(i, class_iou[i])


if __name__ == "__main__":
  app.run(main)
