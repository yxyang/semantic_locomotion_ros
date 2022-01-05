"""Collection of network optimizers."""
import logging

from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop

logger = logging.getLogger("ptsemseg")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(cfg):
  """Gets optimizer from config."""
  if cfg["training"]["optimizer"] is None:
    logger.info("Using SGD optimizer")
    return SGD

  else:
    opt_name = cfg["training"]["optimizer"]["name"]
    if opt_name not in key2opt:
      raise NotImplementedError(
          "Optimizer {} not implemented".format(opt_name))

    logger.info("Using {} optimizer".format(opt_name))
    return key2opt[opt_name]