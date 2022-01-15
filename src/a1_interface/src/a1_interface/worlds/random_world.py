"""Build a world with uneven terrains."""
import numpy as np
from a1_interface.worlds import plane_world
from a1_interface.worlds import soft_world
from a1_interface.worlds import uneven_world


class RandomWorld:
  """Builds a simple world with a plane only."""
  def __init__(self, pybullet_client):
    self._worlds = [
        plane_world.PlaneWorld(pybullet_client),
        soft_world.SoftWorld(pybullet_client),
        uneven_world.UnevenWorld(pybullet_client)
    ]

  def build_world(self):
    world = np.random.choice(self._worlds)
    return world.build_world()
