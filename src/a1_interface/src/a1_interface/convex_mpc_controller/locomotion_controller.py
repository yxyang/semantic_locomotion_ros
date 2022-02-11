"""A model based controller framework."""
from datetime import datetime
from multiprocessing import Process
import os
import pickle
import threading
import time
from typing import Tuple

import ml_collections
import numpy as np
import rospkg
import rospy
from pybullet_utils import bullet_client
import pybullet

from a1_interface.msg import gait_type
from a1_interface.msg import controller_mode
from a1_interface.convex_mpc_controller import com_velocity_estimator
from a1_interface.convex_mpc_controller import offset_gait_generator
from a1_interface.convex_mpc_controller import raibert_swing_leg_controller
from a1_interface.convex_mpc_controller import torque_stance_leg_controller_mpc
from a1_interface.convex_mpc_controller.gait_configs import slow
from a1_interface.robots import a1
from a1_interface.robots import a1_robot
from a1_interface.robots.motors import MotorCommand
from a1_interface.robots.motors import MotorControlMode
from a1_interface.worlds import abstract_world, plane_world


def get_sim_conf():
  config = ml_collections.ConfigDict()
  config.timestep: float = 0.002
  config.action_repeat: int = 1
  config.reset_time_s: float = 3.
  config.num_solver_iterations: int = 30
  config.init_position: Tuple[float, float, float] = (0., 0., 0.32)
  config.init_rack_position: Tuple[float, float, float] = [0., 0., 1]
  config.on_rack: bool = False
  return config


def _dump_data(data, logdir):
  # data, logdir = args
  print("Dump started")
  pickle.dump(data, open(logdir, 'wb'))
  print("Dump done")


class LocomotionController:
  """Generates the quadruped locomotion.

  The actual effect of this controller depends on the composition of each
  individual subcomponent.

  """
  def __init__(
      self,
      use_real_robot: bool = False,
      show_gui: bool = False,
      logdir: str = 'logs/',
      world_class: abstract_world.AbstractWorld = plane_world.PlaneWorld,
      start_running_immediately: bool = True):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the leg swing/stance pattern.
      state_estimator: Estimates the state of the robot (e.g. center of mass
        position or velocity that may not be observable from sensors).
      swing_leg_controller: Generates motor actions for swing legs.
      stance_leg_controller: Generates motor actions for stance legs.
      clock: A real or fake clock source.
    """
    self._use_real_robot = use_real_robot
    self._show_gui = show_gui
    self._world_class = world_class
    self.setup_pybullet_client()
    self.setup_robot_and_controllers()
    self.reset_robot()
    self.reset_controllers()
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._last_command_timestamp = 0
    self._last_image_embedding = np.zeros(4)
    self._logs = []
    if logdir:
      self._logdir = os.path.join(rospkg.get_ros_home(), logdir)
      if not os.path.exists(self._logdir):
        os.makedirs(self._logdir)
      rospy.loginfo("Logging to: {}".format(self._logdir))
    else:
      rospy.loginfo("Logging disabled.")
      self._logdir = None

    self._mode = controller_mode.DOWN
    self.set_controller_mode(controller_mode(mode=controller_mode.STAND))
    slow_gait = slow.get_config()
    self._desired_gait = gait_type(
        step_frequency=slow_gait.gait_parameters[0],
        foot_clearance=slow_gait.foot_clearance_max,
        base_height=slow_gait.desired_body_height,
        max_forward_speed=slow_gait.max_forward_speed,
        recommended_forward_speed=slow_gait.max_forward_speed,
        timestamp=rospy.get_rostime())
    self._gait = self._desired_gait
    self._handle_gait_switch()

    if start_running_immediately:
      self.run_thread = threading.Thread(target=self.run)
      self.run_thread.start()
      # Callback timer
      rospy.Timer(rospy.Duration(1), self._flush_logging)

  def setup_pybullet_client(self):
    if self._show_gui and not self._use_real_robot:
      p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
      p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    self.pybullet_client = p

  def setup_robot_and_controllers(self):
    """Sets up simulated / real robot and corresponding MPC controllers."""
    # Construct robot
    p = self.pybullet_client
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    rp = rospkg.RosPack()
    package_path = rp.get_path('a1_interface')
    p.setAdditionalSearchPath(os.path.join(package_path, 'data'))

    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.002)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    self._world_builder = self._world_class(self.pybullet_client)
    self.ground_id = self._world_builder.build_world()

    # Construct robot class:
    if self._use_real_robot:
      self._robot = a1_robot.A1Robot(
          pybullet_client=p,
          sim_conf=get_sim_conf(),
          motor_control_mode=MotorControlMode.HYBRID)
    else:
      self._robot = a1.A1(pybullet_client=p,
                          sim_conf=get_sim_conf(),
                          motor_control_mode=MotorControlMode.HYBRID)

    if self._show_gui and not self._use_real_robot:
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    self._clock = lambda: self._robot.time_since_reset

    self._gait_generator = offset_gait_generator.OffsetGaitGenerator(
        self._robot, [0., np.pi, np.pi, 0.])

    desired_speed, desired_twisting_speed = (0., 0., 0.), 0.

    self._state_estimator = com_velocity_estimator.COMVelocityEstimator(
        self._robot, velocity_window_size=60, ground_normal_window_size=10)

    self._swing_controller = \
      raibert_swing_leg_controller.RaibertSwingLegController(
          self._robot,
          self._gait_generator,
          self._state_estimator,
          desired_speed=desired_speed,
          desired_twisting_speed=desired_twisting_speed,
          desired_body_height=self._robot.mpc_body_height,
          foot_landing_clearance=0.01,
          foot_height=0.1,
          use_raibert_heuristic=True)

    mpc_friction_coef = 0.4
    self._stance_controller = \
      torque_stance_leg_controller_mpc.TorqueStanceLegController(
          self._robot,
          self._gait_generator,
          self._state_estimator,
          desired_speed=(desired_speed[0], desired_speed[1]),
          desired_twisting_speed=desired_twisting_speed,
          desired_body_height=self._robot.mpc_body_height,
          body_mass=self._robot.mpc_body_mass,
          body_inertia=self._robot.mpc_body_inertia,
          friction_coeffs=np.ones(4) * mpc_friction_coef)

  @property
  def swing_controller(self):
    return self._swing_controller

  @property
  def stance_controller(self):
    return self._stance_controller

  @property
  def gait_generator(self):
    return self._gait_generator

  @property
  def state_estimator(self):
    return self._state_estimator

  @property
  def time_since_reset(self):
    return self._time_since_reset

  def reset_robot(self):
    self._robot.reset(hard_reset=False)
    if self._show_gui and not self._use_real_robot:
      self.pybullet_client.configureDebugVisualizer(
          self.pybullet_client.COV_ENABLE_RENDERING, 1)

  def reset_controllers(self):
    # Resetting other components
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._gait_generator.reset()
    self._state_estimator.reset(self._time_since_reset)
    self._swing_controller.reset(self._time_since_reset)
    self._stance_controller.reset(self._time_since_reset)

  def update(self):
    self._time_since_reset = self._clock() - self._reset_time
    self._gait_generator.update()
    self._state_estimator.update(self._gait_generator.desired_leg_state)
    self._swing_controller.update(self._time_since_reset)
    future_contact_estimate = self._gait_generator.get_estimated_contact_states(
        torque_stance_leg_controller_mpc.PLANNING_HORIZON_STEPS,
        torque_stance_leg_controller_mpc.PLANNING_TIMESTEP)
    self._stance_controller.update(self._time_since_reset,
                                   future_contact_estimate)

  def get_action(self):
    """Returns the control ouputs (e.g. positions/torques) for all motors."""
    swing_action = self._swing_controller.get_action()
    stance_action, qp_sol = self._stance_controller.get_action()

    actions = []
    for joint_id in range(self._robot.num_motors):
      if joint_id in swing_action:
        actions.append(swing_action[joint_id])
      else:
        assert joint_id in stance_action
        actions.append(stance_action[joint_id])

    vectorized_action = MotorCommand(
        desired_position=[action.desired_position for action in actions],
        kp=[action.kp for action in actions],
        desired_velocity=[action.desired_velocity for action in actions],
        kd=[action.kd for action in actions],
        desired_extra_torque=[
            action.desired_extra_torque for action in actions
        ])

    return vectorized_action, dict(qp_sol=qp_sol)

  def _get_stand_action(self):
    return MotorCommand(
        desired_position=self._robot.motor_group.init_positions,
        kp=self._robot.motor_group.kps,
        desired_velocity=0,
        kd=self._robot.motor_group.kds,
        desired_extra_torque=0)

  def _handle_mode_switch(self):
    """Handles robot mode switch based on user command."""
    if self._mode == self._desired_mode:
      return
    self._mode = self._desired_mode
    if self._desired_mode == controller_mode.DOWN:
      rospy.loginfo("Entering joint damping mode.")
      self._flush_logging(None)
    elif self._desired_mode == controller_mode.STAND:
      rospy.loginfo("Standing up.")
      self.reset_robot()
    else:
      rospy.loginfo("Walking.")
      self.reset_controllers()
      self._start_logging()

  def _start_logging(self):
    self._logs = []

  def _update_logging(self, action, qp_sol):
    """Adds latest robot state to logs."""
    frame = dict(
        desired_speed=(self._swing_controller.desired_speed,
                       self._swing_controller.desired_twisting_speed),
        gait_type=self._gait,
        timestamp=self._time_since_reset,
        base_position_ground_frame=self._state_estimator.
        com_position_ground_frame,
        base_quat_ground_frame=self._state_estimator.
        com_orientation_quat_ground_frame,
        base_rpy=self._robot.base_orientation_rpy,
        motor_angles=self._robot.motor_angles,
        base_vel=self._robot.motor_velocities,
        base_vels_body_frame=self._state_estimator.com_velocity_body_frame,
        base_rpy_rate=self._robot.base_rpy_rate,
        motor_vels=self._robot.motor_velocities,
        motor_torques=self._robot.motor_torques,
        foot_contacts=self._robot.foot_contacts,
        desired_grf=qp_sol,
        robot_action=action,
        gait_generator_phase=self._gait_generator.current_phase.copy(),
        gait_generator_state=self._gait_generator.leg_state,
        ground_orientation=self._state_estimator.
        ground_orientation_world_frame,
        foot_velocities=self._robot.foot_velocities,
        foot_forces=self._robot.foot_forces,
        image_embedding=self._last_image_embedding)
    if self._logdir:
      self._logs.append(frame)

  def _flush_logging(self, event):
    """Flush logging to disk if current log object is too big."""
    del event  # unused
    if self._logdir and len(self._logs) > 1000:
      logs = self._logs
      self._logs = []
      filename = 'log_{}.pkl'.format(
          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
      p = Process(target=_dump_data,
                  args=(logs, os.path.join(self._logdir, filename)))
      p.start()
      # pickle.dump(logs, open(os.path.join(self._logdir, filename), 'wb'))
      rospy.loginfo("Data logged to: {}".format(
          os.path.join(self._logdir, filename)))

  def _handle_gait_switch(self):
    """Handles gait switch commands and update corresponding controllers."""
    if (self._gait.step_frequency == self._desired_gait.step_frequency) and (
        self._gait.foot_clearance == self._desired_gait.foot_clearance) and (
            self._gait.base_height == self._desired_gait.base_height):
      return

    self._gait = self._desired_gait
    self._gait_generator.gait_params[0] = self._gait.step_frequency
    self._swing_controller.foot_height = self._gait.foot_clearance
    # self._swing_controller.foot_landing_clearance = \
    #   self._gait_config.foot_clearance_land
    self._swing_controller.desired_body_height = \
      self._gait.base_height
    # self._stance_controller.update_mpc_config(
    #     self._gait_config.mpc_foot_friction, self._gait_config.mpc_body_mass,
    #     self._gait_config.mpc_body_inertia, self._gait_config.mpc_weight)
    self._stance_controller.desired_body_height = self._gait.base_height

  def run(self):
    """The low-level control thread."""
    rospy.loginfo("Low level thread started...")
    while True:
      self._handle_mode_switch()
      self._handle_gait_switch()
      self.update()
      if self._mode == controller_mode.DOWN:
        time.sleep(0.1)
      elif self._mode == controller_mode.STAND:
        action = self._get_stand_action()
        self._robot.step(action)
        time.sleep(0.001)
      elif self._mode == controller_mode.WALK:
        action, qp_sol = self.get_action()
        self._robot.step(action)
        self._update_logging(action, qp_sol)
      else:
        rospy.loginfo("Running loop terminated, exiting...")
        break

      if self._show_gui:
        self.pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=30 + self._robot.base_orientation_rpy[2] / np.pi * 180,
            cameraPitch=-30,
            cameraTargetPosition=self._robot.base_position,
        )

  def __del__(self):
    self.close()

  def close(self):
    if hasattr(self, 'pybullet_client'):
      self.pybullet_client.disconnect()
      del self.pybullet_client
      del self._world_builder

  def set_controller_mode(self, command):
    if not self.is_safe:
      self._desired_mode = controller_mode.DOWN
    else:
      self._desired_mode = command.mode
    self._last_command_timestamp = self._time_since_reset

  def set_gait(self, command):
    self._desired_gait = command

  @property
  def is_safe(self):
    """Computes whether the robot is safe based on height and orientation."""
    if self.mode != controller_mode.WALK:
      return True
    rot_mat = np.array(
        self._robot.pybullet_client.getMatrixFromQuaternion(
            self._state_estimator.com_orientation_quat_ground_frame)).reshape(
                (3, 3))
    up_vec = rot_mat[2, 2]
    base_height = self._robot.base_position[2]
    return up_vec > 0.85 and base_height > 0.14

  @property
  def robot(self):
    return self._robot

  @property
  def mode(self):
    return self._mode

  @property
  def last_command_timestamp(self):
    return self._last_command_timestamp

  @property
  def gait(self):
    return self._gait

  def set_desired_speed(self, speed_command):
    """Sets the robot's desired speed after exponential smoothing.."""
    old_speed = np.array([
        self._swing_controller.desired_speed[0],
        self._swing_controller.desired_speed[1],
        self._swing_controller.desired_twisting_speed
    ])
    new_speed = np.array([
        self._gait.max_forward_speed * speed_command.vel_x,
        0.5 * speed_command.vel_y, 0.5 * speed_command.rot_z
        # self._gait_config.max_side_speed * speed_command.vel_y,
        # self._gait_config.max_rot_speed * speed_command.rot_z
    ])
    smoothed_new_speed = 0.5 * old_speed + 0.5 * new_speed

    desired_lin_speed = (
        smoothed_new_speed[0],
        smoothed_new_speed[1],
        0,
    )
    desired_rot_speed = smoothed_new_speed[2]
    self._swing_controller.desired_speed = desired_lin_speed
    self._swing_controller.desired_twisting_speed = desired_rot_speed
    self._stance_controller.desired_speed = desired_lin_speed
    self._stance_controller.desired_twisting_speed = desired_rot_speed

  def set_image_embedding(self, msg):
    self._last_image_embedding = np.array(msg.embedding)
