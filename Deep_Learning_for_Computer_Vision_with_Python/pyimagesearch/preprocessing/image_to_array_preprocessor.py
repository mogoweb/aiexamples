from keras.preprocessing.image import img_to_array
from keras import backend as K


class ImageToArrayPreprocessor:
  def __init__(self, data_format=None):
    if data_format is None:
      data_format = K.image_data_format()

    self.data_format = data_format


  def preprocess(self, image):
    # apply the Keras utility function that correctly rearranges
    # the dimensions of the image

    return img_to_array(image, data_format=self.data_format)