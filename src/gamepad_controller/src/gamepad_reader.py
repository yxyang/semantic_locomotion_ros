#!/usr/bin/env python
"""Gamepad reader example."""
from absl import app
import time

from gamepad_reader_lib import Gamepad


def main(_):
  gamepad = Gamepad()
  while True:
    speed_command = gamepad.speed_command
    print("Vx: {}, Vy: {}, Wz: {}, Estop: {}".format(speed_command.vel_x,
                                                     speed_command.vel_y,
                                                     speed_command.rot_z,
                                                     gamepad.estop_flagged))
    time.sleep(0.1)
    if gamepad.estop_flagged:
      break
  gamepad.stop()


if __name__ == "__main__":
  app.run(main)
