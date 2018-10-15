import tensorflow as tf
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants

with tf.Graph().as_default() as g1:
  base64_str = tf.placeholder(tf.string, name='input_string')
  input_str = tf.decode_base64(base64_str)
  decoded_image = tf.image.decode_png(input_str, channels=1)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([28, 28])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  # 展开为1维数组
  resized_image_1d = tf.reshape(resized_image, (-1, 28 * 28))
  print(resized_image_1d.shape)
  tf.identity(resized_image_1d, name="DecodeJPGOutput")

g1def = g1.as_graph_def()

with tf.Graph().as_default() as g2:
  with tf.Session(graph=g2) as sess:
    input_graph_def = saved_model_utils.get_meta_graph_def(
        "./model", tag_constants.SERVING).graph_def

    tf.saved_model.loader.load(sess, ["serve"], "./model")

    g2def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        ["myOutput"],
        variable_names_whitelist=None,
        variable_names_blacklist=None)

with tf.Graph().as_default() as g_combined:
  with tf.Session(graph=g_combined) as sess:

    x = tf.placeholder(tf.string, name="base64_input")

    y, = tf.import_graph_def(g1def, input_map={"input_string:0": x}, return_elements=["DecodeJPGOutput:0"])

    z, = tf.import_graph_def(g2def, input_map={"myInput:0": y}, return_elements=["myOutput:0"])
    tf.identity(z, "myOutput")

    tf.saved_model.simple_save(sess,
              "./modelbase64",
              inputs={"base64_input": x},
              outputs={"myOutput": z})
