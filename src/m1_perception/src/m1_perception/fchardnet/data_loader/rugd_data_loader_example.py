"""Example of using data loader from RUGD dataset."""
from absl import app
from absl import flags

from m1_perception.fchardnet.data_loader import rugd_data_loader

flags.DEFINE_string('root_dir', 'fchardnet/data/RUGD', 'dir for RUGD dataset.')
flags.DEFINE_integer('batch_size', 10, 'batch size.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  data_loader = rugd_data_loader.RUGDDataLoader(root_dir=FLAGS.root_dir,
                                                batch_size=FLAGS.batch_size)

  for images, masks in data_loader.data_stream():
    print("Image tensor shape: {}, Mask tensor shape: {}".format(
        images.shape, masks.shape))


if __name__ == "__main__":
  app.run(main)
