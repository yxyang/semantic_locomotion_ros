"""Config for FCHarDNet on RUGD."""
from ml_collections import ConfigDict

from fchardnet.data_loader.augmentations import RandomCrop, GaussianBlur, AdjustGamma, AdjustSaturation, AdjustHue, AdjustBrightness, AdjustContrast, RandomHorizontallyFlip


def get_config():
  """Configs for training on RUGD dataset."""
  config = ConfigDict()

  data = ConfigDict()
  data.data_path = 'fchardnet/data/RUGD'
  data.augmentations = [
      RandomCrop, GaussianBlur, AdjustGamma, AdjustSaturation, AdjustHue,
      AdjustBrightness, AdjustContrast, RandomHorizontallyFlip
  ]
  config.data = data

  training = ConfigDict()
  training.batch_size = 10
  training.num_epoches = 100
  training.eval_frequency = 1
  training.save_frequency = 1
  config.training = training

  config.logdir = 'logs'

  return config
