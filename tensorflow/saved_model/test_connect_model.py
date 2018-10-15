import tensorflow as tf
import numpy as np
import base64

with tf.Session(graph=tf.Graph()) as sess:
  sess.run(tf.global_variables_initializer())

  tf.saved_model.loader.load(sess, ["serve"], "./modelbase64")
  graph = tf.get_default_graph()

  with open("./5.png", "rb") as image_file:
    encoded_string = str(base64.urlsafe_b64encode(image_file.read()), "utf-8")

  x = sess.graph.get_tensor_by_name('base64_input:0')
  y = sess.graph.get_tensor_by_name('myOutput:0')

  scores = sess.run(y,
           feed_dict={x: encoded_string})
  print("predict: %d, actual: %d" % (np.argmax(scores, 1), 5))