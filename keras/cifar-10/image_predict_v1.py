from keras.models import model_from_json
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop

BATCH_SIZE = 128
NB_CLASSES = 10
VERBOSE = 1
OPTIM = RMSprop()

# load json and create model
json_file = open('cifar10_arch_v1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("cifar10_weights_v1.h5")
print("Loaded model from disk")

(_, _), (X_test, y_test) = cifar10.load_data()
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
X_test = X_test.astype('float32')
X_test = X_test / 255

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))