"""Base class for all robots."""
from typing import Any
from typing import Sequence
from typing import Tuple

import ml_collections
import numpy as np

from a1_interface.robots.motors import MotorControlMode
from a1_interface.robots.motors import MotorGroup
from a1_interface.robots.motors import MotorModel
from a1_interface.robots.robot import Robot

_PYBULLET_DEFAULT_PROJECTION_MATRIX = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, -1.0000200271606445, -1.0,
                                       0.0, 0.0, -0.02000020071864128, 0.0)
_DEFAULT_TARGET_DISTANCE = 10


def create_camera_image(pybullet_client,
                        camera_position,
                        camera_orientation,
                        resolution,
                        projection_mat,
                        egl_render=False):
  """Returns synthetic camera image from pybullet."""
  orientation_mat = pybullet_client.getMatrixFromQuaternion(camera_orientation)

  # The first column in the orientation matrix.
  forward_vec = orientation_mat[::3]
  target_distance = _DEFAULT_TARGET_DISTANCE
  camera_target = [
      camera_position[0] + forward_vec[0] * target_distance,
      camera_position[1] + forward_vec[1] * target_distance,
      camera_position[2] + forward_vec[2] * target_distance
  ]

  # The third column in the orientation matrix. We assume camera up vector is
  # always [0, 0, 1] in its local frame.
  up_vec = orientation_mat[2::3]

  view_mat = pybullet_client.computeViewMatrix(camera_position, camera_target,
                                               up_vec)
  renderer = (pybullet_client.ER_BULLET_HARDWARE_OPENGL
              if egl_render else pybullet_client.ER_TINY_RENDERER)
  return pybullet_client.getCameraImage(resolution[0],
                                        resolution[1],
                                        viewMatrix=view_mat,
                                        projectionMatrix=projection_mat,
                                        renderer=renderer)


class A1(Robot):
  """A1 Robot."""
  def __init__(
      self,
      pybullet_client: Any = None,
      sim_conf: ml_collections.ConfigDict = None,
      urdf_path: str = "a1.urdf",
      base_joint_names: Tuple[str, ...] = (),
      foot_joint_names: Tuple[str, ...] = (
          "FR_toe_fixed",
          "FL_toe_fixed",
          "RR_toe_fixed",
          "RL_toe_fixed",
      ),
      motor_control_mode: MotorControlMode = MotorControlMode.POSITION,
      mpc_body_height: float = 0.26,
      mpc_body_mass: float = 110 / 9.8,
      mpc_body_inertia: Tuple[float] = np.array(
          (0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 10.,
  ) -> None:
    """Constructs an A1 robot and resets it to the initial states.
        Initializes a tuple with a single MotorGroup containing 12 MotoroModels.
        Each MotorModel is by default configured for the parameters of the A1.
        """
    motors = MotorGroup((
        MotorModel(
            name="FR_hip_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.0,
            min_position=-0.802851455917,
            max_position=0.802851455917,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="FR_upper_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="FR_lower_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="FL_hip_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.0,
            min_position=-0.802851455917,
            max_position=0.802851455917,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="FL_upper_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="FL_lower_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RR_hip_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.0,
            min_position=-0.802851455917,
            max_position=0.802851455917,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="RR_upper_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RR_lower_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RL_hip_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.0,
            min_position=-0.802851455917,
            max_position=0.802851455917,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="RL_upper_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RL_lower_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
    ))
    self._mpc_body_height = mpc_body_height
    self._mpc_body_mass = mpc_body_mass
    self._mpc_body_inertia = mpc_body_inertia

    super().__init__(
        pybullet_client=pybullet_client,
        sim_conf=sim_conf,
        urdf_path=urdf_path,
        motors=motors,
        base_joint_names=base_joint_names,
        foot_joint_names=foot_joint_names,
    )

  @property
  def mpc_body_height(self):
    return self._mpc_body_height

  @mpc_body_height.setter
  def mpc_body_height(self, mpc_body_height: float):
    self._mpc_body_height = mpc_body_height

  @property
  def mpc_body_mass(self):
    return self._mpc_body_mass

  @mpc_body_mass.setter
  def mpc_body_mass(self, mpc_body_mass: float):
    self._mpc_body_mass = mpc_body_mass

  @property
  def mpc_body_inertia(self):
    return self._mpc_body_inertia

  @mpc_body_inertia.setter
  def mpc_body_inertia(self, mpc_body_inertia: Sequence[float]):
    self._mpc_body_inertia = mpc_body_inertia

  @property
  def hip_positions_in_base_frame(self):
    return (
        (0.17, -0.135, 0),
        (0.17, 0.13, 0),
        (-0.195, -0.135, 0),
        (-0.195, 0.13, 0),
    )

  @property
  def num_motors(self):
    return 12

  @property
  def foot_forces(self):
    return np.zeros(4)

  def get_camera_image(self,
                       resolution=(640, 360),
                       egl_render=False,
                       camera_position=(0.22, 0, 0),
                       camera_orientation_rpy=(0., 0.3, 0.)):
    """Returns synthetic on-robot camera image."""
    p = self.pybullet_client
    camera_pos_relative = camera_position
    camera_orientation_relative = p.getQuaternionFromEuler(
        camera_orientation_rpy)
    transform = p.multiplyTransforms(self.base_position,
                                     self.base_orientation_quat,
                                     camera_pos_relative,
                                     camera_orientation_relative)
    return create_camera_image(
        p,
        camera_position=transform[0],
        camera_orientation=transform[1],
        resolution=resolution,
        projection_mat=_PYBULLET_DEFAULT_PROJECTION_MATRIX,
        egl_render=egl_render)[2]
