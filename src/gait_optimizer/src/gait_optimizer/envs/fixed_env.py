"""Creates a fixed environment for gait optimization."""

from typing import Optional, Sequence

import numpy as np

from a1_interface.convex_mpc_controller import locomotion_controller
from a1_interface.worlds import abstract_world
from a1_interface.msg import controller_mode
from gait_optimizer.envs import metrics


def get_default_gait_config():
  return dict(
      gait_params=[2, np.pi, np.pi, 0, 0.5],  # Trotting with variable freq
      foot_clearance_max=0.1,
      foot_clearance_land=0.01,
      desired_body_height=0.26,
      mpc_foot_friction=0.45,
      mpc_body_mass=110 / 9.8,
      mpc_body_inertia=np.array((0.057, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5.,
      mpc_weight=(1., 1., 0, 0, 0, 20, 0., 0., .1, .1, .1, .0, 0),
  )


def generate_speed_profile(max_speed, acc=1):
  def get_desired_speed(time_since_reset):
    return np.minimum(max_speed, time_since_reset * acc), 0, 0

  return get_desired_speed


class FixedEnv:
  """An environment with fixed terrain properties."""
  def __init__(self,
               world_class: abstract_world.AbstractWorld,
               gait_config: Optional[dict] = None,
               show_gui: bool = False,
               use_real_robot: bool = False,
               episode_length: float = 2):
    if gait_config is None:
      gait_config = get_default_gait_config()
    self._controller = locomotion_controller.LocomotionController(
        use_real_robot=use_real_robot,
        show_gui=show_gui,
        world_class=world_class,
        start_running_immediately=False)
    self._controller.set_controller_mode(
        controller_mode(mode=controller_mode.WALK))
    self._controller._handle_mode_switch()
    self._gait_config = gait_config
    self._episode_length = episode_length
    self._show_gui = show_gui
    self._use_real_robot = use_real_robot

    self._controller.gait_generator.gait_params = gait_config["gait_params"]
    self._controller.swing_controller.foot_height = gait_config[
        "foot_clearance_max"]
    self._controller.swing_controller.foot_landing_clearance = gait_config[
        "foot_clearance_land"]
    self._controller.swing_controller.desired_body_height = gait_config[
        "desired_body_height"]
    self._controller.stance_controller.update_mpc_config(
        gait_config["mpc_foot_friction"], gait_config["mpc_body_mass"],
        gait_config["mpc_body_inertia"], gait_config["mpc_weight"],
        gait_config["desired_body_height"])

  def eval_parameters(self, parameters: Sequence[float]) -> float:
    """Evaluates the parameter and returns the total reward."""
    if self._use_real_robot:
      input("Press Enter to start the next episode...")
    self._controller.gait_generator.gait_params = [
        parameters[0], np.pi, np.pi, 0, 0.5
    ]
    self._controller.swing_controller.foot_height = parameters[1]
    self._controller.swing_controller.desired_body_height = parameters[2]
    self._controller.stance_controller.update_mpc_config(
        self._gait_config["mpc_foot_friction"],
        self._gait_config["mpc_body_mass"],
        self._gait_config["mpc_body_inertia"], self._gait_config["mpc_weight"],
        parameters[2])
    get_desired_speed = generate_speed_profile(parameters[3])

    self._controller.reset_robot()
    self._controller.reset_controllers()
    states = []
    robot = self._controller.robot

    while self._controller.time_since_reset < self._episode_length:
      # Set desired speed and update controller
      desired_speed = get_desired_speed(self._controller.time_since_reset)
      self._controller.swing_controller.desired_speed = desired_speed
      self._controller.stance_controller.desired_speed = desired_speed
      self._controller.update()

      # Step controller
      action, _ = self._controller.get_action()
      robot.step(action)

      # Logging
      states.append(
          dict(is_safe=self._controller.is_safe,
               controller_mode=self._controller.mode,
               gait_type=self._controller.gait,
               timestamp=self._controller.time_since_reset,
               base_velocity=robot.base_velocity,
               base_orientation_rpy=robot.base_orientation_rpy,
               base_rpy_rate=robot.base_rpy_rate,
               motor_angles=robot.motor_angles,
               motor_velocities=robot.motor_velocities,
               motor_torques=robot.motor_torques,
               foot_contacts=robot.foot_contacts))
      if self._show_gui:
        self._controller.pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=30 + robot.base_orientation_rpy[2] / np.pi * 180,
            cameraPitch=-30,
            cameraTargetPosition=robot.base_position,
        )
      if not self._controller.is_safe:
        break

    safety_score = metrics.safety_metric(states)
    energy_score = metrics.energy_metric(states)
    stability_score = metrics.stability_metric(states)
    speed_score = metrics.speed_metric(states)
    return safety_score / 500. - stability_score * 10 - \
      energy_score * 1e-4 + speed_score * 3

  def close(self):
    self._controller.close()
