"""Configs for crawling gait."""
import ml_collections
import numpy as np


def get_config():
  """Configurations for crawling gait."""
  config = ml_collections.ConfigDict()
  config.max_forward_speed = 0.5
  config.max_side_speed = 0.4
  config.max_rot_speed = 0.6

  config.gait_parameters = [1., np.pi, np.pi * 0.5, np.pi * 1.5, 0.26]

  # MPC-related settings
  config.mpc_foot_friction = .99#0.55
  config.mpc_body_mass = 110 / 9.8
  config.mpc_body_inertia = np.array(
      (0.027, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5.
  config.mpc_weight = (1., 1., 0, 0, 0, 10, 0., 0., .1, .3, .3, .0, 0)

  # Swing foot settings
  config.foot_clearance_max = 0.15
  config.foot_clearance_land = -0.01
  return config
