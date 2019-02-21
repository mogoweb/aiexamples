import numpy as np
import cv2


class CropPreprocessor:
  def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
    self.width = width
    self.height = height
    self.horiz = horiz
    self.inter = inter


  def preprocess(self, image):
    crops = []

    # grab the width and height of the image then use these dimensions to
    # define the corners of the image based
    (h, w) = image[:2]
    coords = [
      [0, 0, self,width, self.height],
      [w - self.width, 0, w, self.height],
      [w - self.width, h - self.height, w, h],
      [0, h - self.height, self.width, h]
    ]

    # compute the center crop of the image as well
    dw = int(0.5 * (w - self.width))
    dh = int(0.5 * (h - self.height))
    coords.append([dw, dh, w - dw, h - dh])

    for (startx, starty, endx, endy) in coords:
      crop = image[startx:endx, starty:endy]
      crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
      crops.append(crop)

    if self.horiz:
      # compute the horizontal mirror flips for each crop
      mirrors = [cv2.flip(c, 1) for c in crops]
      crops.extend(mirrors)

    return np.array(crops)