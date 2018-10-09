
import tensorflow as tf
import argparse
import sys


def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.origin_model_dir:
    tf.logging.error('Must set flag --origin_model_dir.')
    return -1

  graph = tf.Graph()
  with graph.as_default():
    with tf.Session() as sess:
      meta_graph_def_to_load = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.origin_model_dir)

      input_bytes = tf.placeholder(tf.string, shape=[], name="input_bytes")
      input_bytes = tf.reshape(input_bytes, [])

      # Transform bitstring to uint8 tensor
      input_tensor = tf.image.decode_png(input_bytes, channels=3)

      # Convert to float32 tensor
      input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.float32)

      # Ensure tensor has correct shape
      input_tensor = tf.reshape(input_tensor, [299, 299, 3])

      # inference function accepts a batch of images
      # So expand the single tensor into a batch of 1
      input_tensor = tf.expand_dims(input_tensor, 0)

      # :TODO(how to insert my input layer?)

      # The input name needs to have "_bytes" suffix.
      inputs = {'image_bytes': input_bytes}
      outputs = {'prediction': graph.get_tensor_by_name('final_result:0')}
      tf.saved_model.simple_save(sess, FLAGS.model_dir, inputs, outputs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      type=str,
      default='',
      help='Where to save the exported graph.'
  )
  parser.add_argument(
      '--origin_model_dir',
      type=str,
      default='',
      help='Folder that contains the original SavedModel.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)