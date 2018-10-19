from keras.models import model_from_json
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import import_pb_to_tensorboard
import tensorflow as tf
import os

# load json and create model
json_file = open('cifar10_arch_v1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("cifar10_weights_v1.h5")
print("Loaded model from disk")


def keras_to_tensorflow(keras_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(keras_model.output[i], out_prefix + str(i + 1))

        sess = K.get_session()

        init_graph = sess.graph.as_graph_def()

        main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

        graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

        if log_tensorboard:
            import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, model_name), output_dir)


output_dir = os.path.join(os.getcwd(), "checkpoint")

keras_to_tensorflow(loaded_model, output_dir=output_dir, model_name="cifar10_v1.pb")
