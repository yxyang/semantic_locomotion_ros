"""Config for FCHarDNet on Cityscape."""
from ml_collections import ConfigDict


def get_config():
  """Configs for rugd training."""
  config = ConfigDict()

  model = ConfigDict()
  model.arch = "hardnet"
  config.model = model

  data = ConfigDict()
  data.dataset = "rugd_a1"
  data.train_split = "train"
  data.val_split = "val"
  data.img_rows = 1024
  data.img_cols = 1024
  data.path = "data/RUGD"
  config.data = data

  training = ConfigDict()
  training.train_iters = 90000
  training.batch_size = 6
  training.val_interval = 500
  training.n_workers = 8
  training.print_interval = 10

  augmentations = ConfigDict()
  augmentations.hflip = 0.5
  augmentations.rscale_crop = [1024, 1024]
  augmentations.brightness = 0.4
  augmentations.contrast = 0.4
  augmentations.saturation = 0.5
  augmentations.hue = 0.1
  augmentations.gamma = 2
  training.augmentations = augmentations

  optimizer = ConfigDict()
  optimizer.name = "sgd"
  optimizer.lr = 0.02
  optimizer.weight_decay = 0.0005
  optimizer.momentum = 0.9
  training.optimizer = optimizer

  loss = ConfigDict()
  loss.name = "bootstrapped_cross_entropy"
  loss.min_k = 4096
  loss.loss_th = 0.3
  loss.size_average = True
  training.loss = loss

  lr_schedule = ConfigDict()
  lr_schedule.name = "poly_lr"
  lr_schedule.max_iter = 90000
  training.lr_schedule = lr_schedule

  training.resume = None
  training.finetune = None
  config.training = training

  return config
