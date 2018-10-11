from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, ["serve"], "./model")
  graph = tf.get_default_graph()

  input = np.expand_dims(mnist.test.images[0], 0)
  x = sess.graph.get_tensor_by_name('myInput:0')
  y = sess.graph.get_tensor_by_name('myOutput:0')
  batch_xs, batch_ys = mnist.test.next_batch(1)
  scores = sess.run(y,
           feed_dict={x: batch_xs})
  print("predict: %d, actual: %d" % (np.argmax(scores, 1), np.argmax(batch_ys, 1)))
