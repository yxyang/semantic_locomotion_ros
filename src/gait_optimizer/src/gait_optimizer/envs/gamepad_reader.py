"""Interface for reading gamepad commands.

Note that this version of gamepad_reader is currently designed for
gait_optimizer only, and is not intended to be used otherwhere. The
gamepad_reader package contains a fully-enabled version of gamepad_reader
that's intended to be used for long-term outdoor operations.
"""

import threading
import time

from gamepad_controller.third_party import inputs

MAX_ABS_VAL = 32768


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
               vel_scale_x: float = 1.,
               vel_scale_y: float = 1.,
               vel_scale_rot: float = 2.):
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
    self._lj_pressed = False
    self.is_running = True

    # Controller states
    self.vx, self.vy, self.wz = 0., 0., 0.
    self.read_thread = threading.Thread(target=self.read_loop)
    self.read_thread.start()

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
    if event.ev_type == 'Key' and event.code == 'BTN_THUMBL':
      self._lj_pressed = bool(event.state)
    elif event.ev_type == 'Absolute' and event.code == 'ABS_RX':
      # Left Joystick L/R axis
      self.vy = _interpolate(-event.state, MAX_ABS_VAL, self._vel_scale_y)
    elif event.ev_type == 'Absolute' and event.code == 'ABS_Y':
      # Left Joystick F/B axis; need to flip sign for consistency
      self.vx = _interpolate(-event.state, MAX_ABS_VAL, self._vel_scale_x)
    elif event.ev_type == 'Absolute' and event.code == 'ABS_X':
      self.wz = _interpolate(-event.state, MAX_ABS_VAL, self._vel_scale_rot)

  def hold_until_lj_is_pressed(self):
    """Keep sending vibration signals until Left joystick is pressed."""
    while True:
      try:
        self.gamepad.set_vibration(1., 1., 200)
      except OSError:
        pass
      start_time = time.time()
      while time.time() - start_time < 1:
        if self._lj_pressed:
          return
        time.sleep(0.01)

  @property
  def speed_command(self):
    """Computes speed command from user input."""
    return [self.vx, self.vy, self.wz]

  @property
  def lj_pressed(self):
    return self._lj_pressed

  def stop(self):
    self.is_running = False
