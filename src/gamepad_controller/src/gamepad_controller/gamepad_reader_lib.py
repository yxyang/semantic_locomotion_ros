"""Interface for reading commands from Logitech F710 Gamepad."""
import enum
import itertools
import threading
import time

from absl import flags

import numpy as np
import rospy
from third_party import inputs

from a1_interface.convex_mpc_controller.gait_configs import slow, mid, fast
from a1_interface.msg import controller_mode
from a1_interface.msg import gait_command
from a1_interface.msg import speed_command

FLAGS = flags.FLAGS
MAX_ABS_VAL = 32768

ALLOWED_MODES = [
    controller_mode.STAND, controller_mode.WALK, controller_mode.DOWN
]

ALLOWED_GAITS = [slow.get_config(), mid.get_config(), fast.get_config()]


class GaitMode(enum.Enum):
  """Set of possible operation modes."""
  # Manually switch between a set of discrete gaits, and manually control
  # robot speed.
  MANUAL_SPEED_MANUAL_GAIT = 1
  # Policy specifies gait. Teleop specifies desired speed / steering.
  MANUAL_SPEED_AUTO_GAIT = 2
  # Policy specifies gait + speed. Teleop specifies steering.
  AUTO_SPEED_AUTO_GAIT = 3
  # Fully autonomous operation
  AUTO_NAV = 4
  # Ablation study: Fixed, constant speed; adaptive gait
  FIXED_SPEED_AUTO_GAIT = 5
  # Ablation study: Fixed, constant speed; non-adaptive gait
  AUTO_SPEED_FIXED_GAIT = 6


def _interpolate(raw_reading, max_raw_reading, new_scale):
  return raw_reading / max_raw_reading * new_scale


class Gamepad:
  """Interface for reading commands from Logitech F710 Gamepad.

  The control works as following:
  1) Press LB+RB at any time for emergency stop
  2) Use the left joystick for forward/backward/left/right walking.
  3) Use the right joystick for rotation around the z-axis.
  """
  def __init__(self,
               vel_scale_x: float = 2.,
               vel_scale_y: float = .6,
               vel_scale_rot: float = 1.,
               max_acc: float = 6.):
    """Initialize the gamepad controller.
    Args:
      vel_scale_x: maximum absolute x-velocity command.
      vel_scale_y: maximum absolute y-velocity command.
      vel_scale_rot: maximum absolute yaw-dot command.
    """
    self.gamepad = inputs.devices.gamepads[0]

    self._vel_scale_x = vel_scale_x
    self._vel_scale_y = vel_scale_y
    self._vel_scale_rot = vel_scale_rot
    self._lb_pressed = False
    self._rb_pressed = False
    self._lj_pressed = False
    self._rj_pressed = False
    self._walk_height = 0.
    self._foot_height = 0.
    self._fixed_speed = 0.

    self._manual_gait_generator = itertools.cycle(ALLOWED_GAITS)
    self._manual_gait = next(self._manual_gait_generator)
    self._mode_generator = itertools.cycle(ALLOWED_MODES)
    self._mode = next(self._mode_generator)
    self._gait_mode_generator = itertools.cycle([
        GaitMode.MANUAL_SPEED_MANUAL_GAIT, GaitMode.MANUAL_SPEED_AUTO_GAIT,
        GaitMode.AUTO_SPEED_AUTO_GAIT, GaitMode.AUTO_NAV,
        GaitMode.FIXED_SPEED_AUTO_GAIT, GaitMode.AUTO_SPEED_FIXED_GAIT
    ])
    self._gait_mode = next(self._gait_mode_generator)
    self._skip_waypoint = False

    # Controller states
    self.vx_raw, self.vy_raw, self.wz_raw = 0., 0., 0.
    self.vx, self.vy, self.wz = 0., 0., 0.
    self._max_acc = max_acc
    self._estop_flagged = False
    self.is_running = True
    self.last_timestamp = time.time()

    self.read_thread = threading.Thread(target=self.read_loop)
    self.read_thread.start()
    print("To confirm that you are using the correct gamepad, press down the "
          "LEFT joystick to continue...")
    while True:
      self.gamepad.set_vibration(1., 1., 200)
      start_time = time.time()
      while time.time() - start_time < 1:
        if self._lj_pressed:
          return
        time.sleep(0.01)

  def read_loop(self):
    """The read loop for events.

    This funnction should be executed in a separate thread for continuous
    event recording.
    """
    while self.is_running:
      try:
        events = self.gamepad.read()
        for event in events:
          self.update_command(event)
      except inputs.UnknownEventCode:
        pass

  def update_command(self, event):
    """Update command based on event readings."""
    if event.ev_type == 'Key' and event.code == 'BTN_TL':
      self._lb_pressed = bool(event.state)
      if not self._estop_flagged and event.state == 0:
        self._manual_gait = next(self._manual_gait_generator)
    elif event.ev_type == 'Key' and event.code == 'BTN_TR':
      self._rb_pressed = bool(event.state)
      if not self._estop_flagged and event.state == 0:
        self._mode = next(self._mode_generator)
        rospy.loginfo("Switched control mode to {}.".format(self._mode))
    elif event.ev_type == 'Key' and event.code == 'BTN_THUMBL':
      self._lj_pressed = bool(event.state)
    elif event.ev_type == 'Key' and event.code == 'BTN_THUMBR':
      self._rj_pressed = bool(event.state)
      if self._rj_pressed:
        self._gait_mode = next(self._gait_mode_generator)
        rospy.loginfo("Switched gait mode to {}.".format(self._gait_mode))
    elif event.ev_type == 'Absolute' and event.code == 'ABS_RX':
      # Left Joystick L/R axis
      self.vy_raw = _interpolate(-event.state, MAX_ABS_VAL, self._vel_scale_y)
    elif event.ev_type == 'Absolute' and event.code == 'ABS_Y':
      # Left Joystick F/B axis; need to flip sign for consistency
      self.vx_raw = _interpolate(-event.state, MAX_ABS_VAL, self._vel_scale_x)
    elif event.ev_type == 'Absolute' and event.code == 'ABS_X':
      self.wz_raw = _interpolate(-event.state, MAX_ABS_VAL,
                                 self._vel_scale_rot)
    elif event.ev_type == 'Absolute' and event.code == 'ABS_HAT0X':
      self._skip_waypoint = bool(event.state)
    elif event.ev_type == 'Absolute' and event.code == 'ABS_HAT0Y':
      delta_speed = -0.1 * event.state
      self._fixed_speed = np.clip(self._fixed_speed + delta_speed, 0, 2)

    if self._estop_flagged and self._lj_pressed:
      self._estop_flagged = False
      rospy.loginfo("Estop Released.")

    if self._lb_pressed and self._rb_pressed:
      if not self._estop_flagged:
        rospy.loginfo("EStop Flagged, press LEFT joystick to release.")
      self._estop_flagged = True
      while self._mode != controller_mode.DOWN:
        self._mode = next(self._mode_generator)
      self.vx_raw, self.vy_raw, self.wz_raw = 0., 0., 0.

  @property
  def speed_command(self):
    """Computes speed command from user input."""
    delta_time = np.minimum(time.time() - self.last_timestamp, 1)
    max_delta_speed = self._max_acc * delta_time
    self.vx = np.clip(self.vx_raw, self.vx - max_delta_speed,
                      self.vx + max_delta_speed)
    self.vy = np.clip(self.vy_raw, self.vy - max_delta_speed,
                      self.vy + max_delta_speed)
    self.wz = np.clip(self.wz_raw, self.wz - max_delta_speed,
                      self.wz + max_delta_speed)

    self.last_timestamp = time.time()
    return speed_command(vel_x=self.vx,
                         vel_y=self.vy,
                         rot_z=self.wz,
                         timestamp=rospy.get_rostime())

  @property
  def estop_flagged(self):
    return self._estop_flagged

  @property
  def gait_mode(self):
    return self._gait_mode

  def flag_estop(self):
    if not self._estop_flagged:
      rospy.loginfo("Estop flagged by program.")
      self._estop_flagged = True
      while self._mode != controller_mode.DOWN:
        self._mode = next(self._mode_generator)

  @property
  def mode_command(self):
    if self._estop_flagged:
      return controller_mode(mode=controller_mode.DOWN,
                             timestamp=rospy.get_rostime())
    else:
      return controller_mode(mode=self._mode, timestamp=rospy.get_rostime())

  @property
  def gait_command(self):
    return gait_command(timing_parameters=self._manual_gait.timing_parameters,
                        foot_clearance=self._manual_gait.foot_clearance_max,
                        base_height=self._manual_gait.desired_body_height,
                        timestamp=rospy.get_rostime())

  @property
  def vel_scale_x(self):
    return self._vel_scale_x

  @property
  def vel_scale_y(self):
    return self._vel_scale_y

  @property
  def vel_scale_rot(self):
    return self._vel_scale_rot

  def stop(self):
    self.is_running = False

  @property
  def skip_waypoint(self):
    return self._skip_waypoint

  @skip_waypoint.setter
  def skip_waypoint(self, val):
    self._skip_waypoint = val

  @property
  def fixed_speed(self):
    return self._fixed_speed
