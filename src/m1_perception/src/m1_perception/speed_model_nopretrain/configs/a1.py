"""Config for FCHarDNet on RUGD."""
from ml_collections import ConfigDict

from m1_perception.speed_model_nopretrain.data_loader.augmentations import GaussianBlur, AdjustGamma, AdjustSaturation, AdjustHue, AdjustBrightness, AdjustContrast


def get_config():
  """Configs for training on RUGD dataset."""
  config = ConfigDict()

  data = ConfigDict()
  data.train_data_path = ('m1_perception/speed_model/data/'
                          'ghost_memory_taylor_demo2/train')
  data.val_data_path = ('m1_perception/speed_model/data/'
                        'ghost_memory_taylor_demo/val')
  data.augmentations = [
      GaussianBlur, AdjustGamma, AdjustSaturation, AdjustHue, AdjustBrightness,
      AdjustContrast
  ]
  config.data = data

  training = ConfigDict()
  training.batch_size = 32
  training.num_epoches = 60
  training.eval_frequency = 1
  training.save_frequency = 1
  config.training = training

  config.logdir = 'speed_model_logs/nopretrain'

  return config
