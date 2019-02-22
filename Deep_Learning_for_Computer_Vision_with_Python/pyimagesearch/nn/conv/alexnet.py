from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras import backend as K


class AlexNet:
  @staticmethod
  def build(width, height, depth, classes, reg=0.0002):
    model = Sequential()
    input_shape = (width, height, depth)
    channel_dim = -1

    if K.image_data_format() == "channels_first":
      input_shape = (depth, width, height)
      channel_dim = 1

    # block #1: CONV => RELU => POOL
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape,
                     padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # block #2: CONV => RELU => POOL
    model.add(Conv2D(256, (5, 5),
                     padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # block #3: CONV => RELU => CONV => RELU => CONV => RELU => POOL
    model.add(Conv2D(384, (3, 3),
                     padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(Conv2D(384, (3, 3),
                     padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(Conv2D(256, (3, 3),
                     padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_dim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # block #4: FC => RELU
    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # block #5: FC => RELU
    model.add(Dense(4096, kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes, kernel_regularizer=l2(reg)))
    model.add(Activation("softmax"))

    return model
