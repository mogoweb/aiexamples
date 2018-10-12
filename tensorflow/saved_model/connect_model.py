import tensorflow as tf
from tensorflow.python.framework import ops

def copy_variable_to_graph(org_instance, to_graph, copied_variables={}):
  """
  Copies the Variable instance 'org_instance' into the graph 'to_graph'.
  The dict 'copied_variables', if provided, will be updated with
  mapping the new variable's name to the instance.
  """

  if not isinstance(org_instance, tf.Variable):
    raise TypeError(str(org_instance) + " is not a Variable")

  # The name of the new variable
  new_name = org_instance.name[:org_instance.name.index(':')]
  print("new_name:", new_name)

  # Get the collections that the new instance needs to be added to.
  # The new collections will also be a part of the given namespace,
  # except the special ones required for variable initialization and
  # training.
  collections = []
  for name, collection in org_instance.graph._collections.items():
    if org_instance in collection:
      if (name == ops.GraphKeys.VARIABLES or
          name == ops.GraphKeys.TRAINABLE_VARIABLES):
        collections.append(name)

  # See if its trainable.
  trainable = (org_instance in org_instance.graph.get_collection(
    ops.GraphKeys.TRAINABLE_VARIABLES))
  # Get the initial value
  with org_instance.graph.as_default():
    temp_session = tf.Session()
    init_value = temp_session.run(org_instance.initialized_value())
    print("init_value:", init_value)

  # Initialize the new variable
  with to_graph.as_default():
    new_var = tf.Variable(init_value,
                          trainable,
                          name=new_name,
                          collections=collections,
                          validate_shape=False)

  # Add to the copied_variables dict
  copied_variables[new_var.name] = new_var

  return new_var

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
    tf.saved_model.loader.load(sess, ["serve"], "./model")

    variables = tf.global_variables()
    for v in variables:
      print("v:", v.name)

g2def = g2.as_graph_def()

with tf.Graph().as_default() as g_combined:
  with tf.Session(graph=g_combined) as sess:

    x = tf.placeholder(tf.string, name="base64_input")

    y, = tf.import_graph_def(g1def, input_map={"input_string:0": x}, return_elements=["DecodeJPGOutput:0"])

    z, = tf.import_graph_def(g2def, input_map={"myInput:0": y}, return_elements=["myOutput:0"])
    tf.identity(z, "myOutput")

    sess.run(tf.global_variables_initializer())

    print("before copy variables:", tf.global_variables())
    copied_variables = {}
    for var in variables:
      copy_variable_to_graph(var, g_combined, copied_variables={})
    print("after copy variables:", tf.global_variables())

    tf.saved_model.simple_save(sess,
              "./modelbase64",
              inputs={"base64_input": x},
              outputs={"myOutput": z})

