#!/usr/bin/env python
"""Web server for camera visualization."""
import functools
import http.server
import os
import socketserver

from absl import app
from absl import flags
from absl import logging

import rospkg

flags.DEFINE_integer('port', 9091, 'port for http server.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  rospack = rospkg.RosPack()
  package_dir = os.path.join(rospack.get_path("info_publisher"), "src",
                             "info_publisher")
  handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                              directory=package_dir)
  with socketserver.TCPServer(("", FLAGS.port), handler) as httpd:
    logging.info("Server started at localhost:{}".format(FLAGS.port))
    httpd.serve_forever()


if __name__ == "__main__":
  app.run(main)
