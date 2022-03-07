"""Uses a separate process to log robot data."""
import atexit
import functools
import multiprocessing
import queue
import sys
import traceback

import rospy

from a1_interface.convex_mpc_controller import data_logger_impl


class DataLogger:
  """Wraps data logger as an independent process."""

  # Message types for communication via the pipe.
  _UPDATE = 1
  _UPDATE_RESULT = 2
  _FLUSH = 3
  _FLUSH_RESULT = 4
  _CLOSE = 5
  _EXCEPTION = 6

  def __init__(self, **kwargs):
    # self._conn, conn = multiprocessing.Pipe()
    self._data_queue = multiprocessing.Queue()
    self._message_queue = multiprocessing.Queue()
    self._process = multiprocessing.Process(
        target=self._worker,
        args=(self._data_queue, self._message_queue,
              data_logger_impl.DataLoggerImpl, kwargs))
    atexit.register(self.close)
    self._process.start()

  def flush_logging(self, blocking=False):
    self._data_queue.put((self._FLUSH, None))
    if blocking:
      return self._receive(self._FLUSH_RESULT)
    else:
      return functools.partial(self._receive, self._FLUSH_RESULT)

  def update_logging(self, new_content, blocking=False):
    self._data_queue.put((self._UPDATE, new_content))
    if blocking:
      return self._receive(self._UPDATE_RESULT)
    else:
      return functools.partial(self._receive, self._UPDATE_RESULT)

  def close(self):
    """Closes the data logger and the related queues."""
    if self._process:
      try:
        self._data_queue.put((self._CLOSE, None))
        self._data_queue.close()
      except IOError:
        # Connection already closed.
        pass
      self._process.join()
      del self._process
      del self._data_queue
      del self._message_queue
      self._data_queue = None
      self._message_queue = None
      self._process = None
    else:
      pass  # Don't close a connection twice

  def _receive(self, expected_message):
    message, payload = self._message_queue.get(block=True)
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == expected_message:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, data_queue, message_queue, constructor,
              constructor_kwargs):
    """Wait for new data from control loop and stores / flushes."""
    try:
      logger = constructor(**constructor_kwargs)
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          message, payload = data_queue.get(block=True, timeout=0.1)
        except (EOFError, KeyboardInterrupt, queue.Empty):
          continue

        if message == self._UPDATE:
          message_queue.put(
              (self._UPDATE_RESULT, logger.update_logging(payload)))
          continue
        if message == self._FLUSH:
          assert payload is None
          message_queue.put((self._FLUSH_RESULT, logger.flush_logging()))
          continue
        if message == self._CLOSE:
          assert payload is None
          logger.close()
          break

        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:  # pylint: disable=broad-except
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      message_queue.put((self._EXCEPTION, stacktrace))
      rospy.loginfo("Error in logging process: {}".format(stacktrace))

    data_queue.close()
    message_queue.close()
