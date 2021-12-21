"""Configs for crawling gait."""
import ml_collections
import numpy as np


def get_config():
  """Congigurations for trotting gait."""
  config = ml_collections.ConfigDict()
  config.max_forward_speed = 1.2
  config.max_side_speed = 0.7
  config.max_rot_speed = 1.

  config.gait_parameters = [2.5, np.pi, np.pi, 0., 0.5]

  # MPC-related settings
  config.mpc_foot_friction = 0.45
  config.mpc_body_mass = 110 / 9.8
  config.mpc_body_inertia = np.array(
      (0.057, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5.
  config.mpc_weight = (1., 1., 0, 0, 0, 10, 0., 0., .1, .1, .1, .0, 0)

  config.desired_body_height = 0.26
  # Swing foot settings
  config.foot_clearance_max = 0.14
  config.foot_clearance_land = -0.01
  return config
