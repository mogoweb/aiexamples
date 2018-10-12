
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import sys

GRAPH1_PATH = "./models/g1/output_graph.pb"
GRAPH2_PATH = "./models/g2/output_graph.pb"
OUTPUT_PATH = "./models/output/output_graph.pb"

def add_jpeg_decoding(input_width, input_height, input_depth):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_width: The image width.
    input_height: The image height.
    input_depth: The image channels.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  tf.identity(resized_image, name="DecodeJPGOutput")
  return jpeg_data, resized_image


def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  # if not FLAGS.origin_model_dir:
  #   tf.logging.error('Must set flag --origin_model_dir.')
  #   return -1

  g1 = tf.Graph()
  with g1.as_default():
    add_jpeg_decoding(299, 299, 3)
    with tf.Session(graph=g1) as sess:
      w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
      w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())
      saver.save(sess=sess, save_path=GRAPH1_PATH)

  with tf.Graph().as_default() as graph:
    input_bytes = tf.placeholder(tf.string, shape=[], name="input_bytes")
    graph1 = tf.train.import_meta_graph(GRAPH1_PATH+".meta", clear_devices=True, input_map={"DecodeJPGInput": input_bytes})
    graph2 = tf.train.import_meta_graph("./models/g2/_retrain_checkpoint.meta", clear_devices=True, input_map={"module/hub_input/images": graph.get_tensor_by_name("DecodeJPGOutput:0")})

    with tf.Session(graph=graph) as sess:
      sess.run(tf.global_variables_initializer())
      # graph1_restorer = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="G1_TRAINABLE_VARIABLE_SCOPE"))
      # graph1_restorer.restore(sess, save_path=GRAPH1_PATH)
      graph2_restorer = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
      graph2_restorer.restore(sess, save_path=GRAPH2_PATH)

      saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_STEP, scope="G1_TRAINABLE_VARIABLE_SCOPE"))
      saver.save(sess=sess, save_path=OUTPUT_PATH)



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
  parser.add_argument(
      '--tfhub_module',
      type=str,
      default=(
          'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),
      help="""\
        Which TensorFlow Hub module to use.
        See https://github.com/tensorflow/hub/blob/master/docs/modules/image.md
        for some publicly available ones.\
        """)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)