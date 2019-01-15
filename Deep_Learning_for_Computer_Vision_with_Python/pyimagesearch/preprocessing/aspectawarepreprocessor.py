import imutils
import cv2


class AspectAwarePreprocessor:
  def __init__(self, width, height, inter=cv2.INTER_AREA):
    # store the target image width, height, and interpolation
    # method used when resing
    self.width = width
    self.height = height
    self.inter = inter

  def preprocess(self, image):
    # grab the dimensions of the image and then initialize
    # the deltas to use when cropping
    (h, w) = image.shape[:2]
    dW = 0
    dH = 0
    # if the width is smaller than the height, then resize
    # along the width (i.e., the smaller dimension) and then update
    # the deltas to crop the height to the desired dimension
    if w < h:
      image = imutils.resize(image, width=self.width, inter=self.inter)
      dH = int((image.shape[0] - self.height) / 2.0)
    else:
      image = imutils.resize(image, height=self.height, inter=self.inter)
      dW = int((image.shape[1] - self.width) / 2.0)

    # now that our images have ben resized, we need to re-grab the width
    # and height, followed by performing the crop
    (h, w) = image.shape[:2]
    image = image[dH:h-dH, dW:w-dW]
    return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
