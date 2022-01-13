"""Creates a fixed environment for gait optimization."""

import pickle
import time
from typing import Optional, Sequence

import numpy as np
import rospy

from a1_interface.convex_mpc_controller import locomotion_controller
from a1_interface.worlds import abstract_world
from a1_interface.msg import controller_mode
from gait_optimizer.envs import gamepad_reader
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


def generate_slowdown_speed_profile(curr_speed, time_to_stop=1):
  start_speed = np.maximum(curr_speed, 0)
  def get_desired_speed(time_since_reset):
    return np.maximum(start_speed * (time_to_stop - time_since_reset), 0), 0, 0

  return get_desired_speed


def clip_swing_freq(parameters, max_swing_distance=0.3):
  clipped_parameters = np.array(parameters).copy()
  max_speed = clipped_parameters[3]
  min_swing_freq = max_speed / (2 * max_swing_distance)
  clipped_parameters[0] = np.maximum(min_swing_freq, clipped_parameters[0])
  return clipped_parameters


class FixedEnv:
  """An environment with fixed terrain properties."""
  def __init__(self,
               world_class: abstract_world.AbstractWorld,
               gait_config: Optional[dict] = None,
               show_gui: bool = False,
               use_real_robot: bool = False,
               episode_length: float = 2,
               settledown_time: float = 2):
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
    self._settledown_time = settledown_time  # For real robot use only.
    self._latest_trajectory = None

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

    if self._use_real_robot:
      self._gamepad = gamepad_reader.Gamepad()

  def eval_parameters(self, parameters: Sequence[float]) -> float:
    """Evaluates the parameter and returns the total reward."""
    parameters = clip_swing_freq(parameters)

    if self._use_real_robot:
      print("Press left joystick on the gamepad to start the next episode...")
      self._gamepad.hold_until_lj_is_pressed()

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
               base_velocity=self._controller.state_estimator.
               com_velocity_body_frame,
               base_orientation_rpy=robot.base_orientation_rpy,
               base_rpy_rate=robot.base_rpy_rate,
               motor_angles=robot.motor_angles,
               motor_velocities=robot.motor_velocities,
               motor_torques=robot.motor_torques,
               foot_contacts=robot.foot_contacts,
               foot_velocities=robot.foot_velocities))
      if self._show_gui:
        self._controller.pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=30 + robot.base_orientation_rpy[2] / np.pi * 180,
            cameraPitch=-30,
            cameraTargetPosition=robot.base_position,
        )
      if not self._controller.is_safe:
        break

    if self._use_real_robot:
      self._slowdown(robot.base_velocity)
      self._reset_with_gamepad()

    safety_score = metrics.safety_metric(states)
    energy_score = metrics.energy_metric(states)
    # stability_score = metrics.stability_metric(states)
    speed_score = metrics.speed_metric(states)
    foot_velocity_score = metrics.foot_velocity_metric(states)
    self._latest_trajectory = states
    return safety_score - foot_velocity_score * 30 - \
      energy_score * 1e-3 + speed_score * 3

  def _slowdown(self, current_speed):
    """Slow down the robot using a default robust gait."""
    if not self._controller.is_safe:
      rospy.loginfo("Robot unsafe, skipping slow-down...")
    self._controller.gait_generator.gait_params = [3.5, np.pi, np.pi, 0, 0.5]
    self._controller.swing_controller.foot_height = 0.1
    self._controller.swing_controller.desired_body_height = 0.26
    self._controller.stance_controller.update_mpc_config(
        self._gait_config["mpc_foot_friction"],
        self._gait_config["mpc_body_mass"],
        self._gait_config["mpc_body_inertia"], self._gait_config["mpc_weight"],
        0.26)
    robot = self._controller.robot
    get_desired_speed = generate_slowdown_speed_profile(current_speed[0])
    while self._controller.time_since_reset < \
      self._episode_length + self._settledown_time:
      desired_speed = get_desired_speed(self._controller.time_since_reset -
                                        self._episode_length)
      self._controller.swing_controller.desired_speed = desired_speed
      self._controller.stance_controller.desired_speed = desired_speed
      self._controller.update()

      # Step controller
      action, _ = self._controller.get_action()
      robot.step(action)
      if not self._controller.is_safe:
        break

  def _reset_with_gamepad(self):
    """Prompts user to steer/move the robot for the next iteration."""
    if not self._controller.is_safe:
      rospy.loginfo("Robot unsafe, skipping reset...")
    self._controller.gait_generator.gait_params = [3, np.pi, np.pi, 0, 0.5]
    self._controller.swing_controller.foot_height = 0.1
    self._controller.swing_controller.desired_body_height = 0.26
    self._controller.stance_controller.update_mpc_config(
        self._gait_config["mpc_foot_friction"],
        self._gait_config["mpc_body_mass"],
        self._gait_config["mpc_body_inertia"], self._gait_config["mpc_weight"],
        0.26)
    robot = self._controller.robot

    while not self._gamepad.lj_pressed:
      speed_command = self._gamepad.speed_command
      desired_lin_speed = (speed_command[0], speed_command[1], 0)
      desired_twisting_speed = speed_command[2]
      self._controller.swing_controller.desired_speed = desired_lin_speed
      self._controller.swing_controller.desired_twisting_speed = \
        desired_twisting_speed
      self._controller.stance_controller.desired_speed = desired_lin_speed
      self._controller.stance_controller.desired_twisting_speed = \
        desired_twisting_speed
      self._controller.update()
      # Step controller
      action, _ = self._controller.get_action()
      robot.step(action)
      if not self._controller.is_safe:
        break

    time.sleep(1)

  def close(self):
    self._controller.close()
    if self._use_real_robot:
      self._gamepad.stop()

  @property
  def latest_trajectory(self):
    return self._latest_trajectory

  def save_latest_trajectory(self, logdir):
    with open(logdir, 'wb') as f:
      pickle.dump(self._latest_trajectory, f)
