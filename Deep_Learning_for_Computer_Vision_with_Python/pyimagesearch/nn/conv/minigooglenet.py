from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K


class MiniGoogleNet:
  @staticmethod
  def conv_module(x, k, kx, ky, stride, channel_dim, padding="same"):
    # define a CONV => BN => RELU pattern
    x = Conv2D(k, (kx, ky), strides=stride, padding=padding)(x)
    x = BatchNormalization(axis=channel_dim)(x)
    x = Activation("relu")(x)

    return x


  @staticmethod
  def inception_module(x, num_k1x1, num_k3x3, channel_dim):
    # define two CONV module, then concatenate across the channel dimension
    conv_1x1 = MiniGoogleNet.conv_module(x, num_k1x1, 1, 1, (1, 1), channel_dim=channel_dim)
    conv_3x3 = MiniGoogleNet.conv_module(x, num_k3x3, 3, 3, (1, 1), channel_dim=channel_dim)
    x = concatenate([conv_1x1, conv_3x3], axis=channel_dim)

    return x


  @staticmethod
  def downsample_module(x, k, channel_dim):
    # define the CONV module and POOL, then concatenate across the channel dimension
    conv_3x3 = MiniGoogleNet.conv_module(x, k, 3, 3, (2, 2), channel_dim=channel_dim, padding="valid")
    pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([conv_3x3, pool], axis=channel_dim)

    return x

  @staticmethod
  def build(width, height, depth, classes):
    input_shape = (width, height, depth)
    channel_dim = -1

    if K.image_data_format() == "channels_first":
      input_shape = (depth, width, height)
      channel_dim = 1

    inputs = Input(shape=input_shape)
    x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), channel_dim=channel_dim)
    x = MiniGoogleNet.inception_module(x, 32, 32, channel_dim=channel_dim)
    x = MiniGoogleNet.inception_module(x, 32, 48, channel_dim=channel_dim)
    x = MiniGoogleNet.downsample_module(x, 80, channel_dim=channel_dim)

    x = MiniGoogleNet.inception_module(x, 112, 48, channel_dim=channel_dim)
    x = MiniGoogleNet.inception_module(x, 96, 64, channel_dim=channel_dim)
    x = MiniGoogleNet.inception_module(x, 80, 80, channel_dim=channel_dim)
    x = MiniGoogleNet.inception_module(x, 48, 96, channel_dim=channel_dim)
    x = MiniGoogleNet.downsample_module(x, 96, channel_dim=channel_dim)

    x = MiniGoogleNet.inception_module(x, 176, 160, channel_dim=channel_dim)
    x = MiniGoogleNet.inception_module(x, 176, 160, channel_dim=channel_dim)
    x = AveragePooling2D((7, 7))(x)
    x = Dropout(0.5)(x)

    # softmax classifier
    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x, name="googlenet")

    return model
