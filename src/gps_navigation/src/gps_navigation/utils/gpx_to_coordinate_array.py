#!/usr/bin/env python
"""Converts GPSX tracking trajectories into array of GPS coordinates."""
from absl import app
from absl import flags

import gpxpy
import numpy as np

flags.DEFINE_string('gpx_file_path', '/path/to/gpx/file', 'path to gpx file.')
flags.DEFINE_string('output_path', '/path/to/output', 'output path.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  gpx_file = open(FLAGS.gpx_file_path, 'r')
  gpx = gpxpy.parse(gpx_file)

  points = []

  for point in gpx.tracks[0].segments[0].points:
    points.append([point.latitude, point.longitude])

  np.save(open(FLAGS.output_path, 'wb'), points)


if __name__ == "__main__":
  app.run(main)
