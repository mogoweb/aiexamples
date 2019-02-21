from sklearn.feature_extraction.image import extract_patches_2d


class PatchProcessor:
  def __init__(self, width, height):
    self.width = width
    self.height = height


  def preprocess(self, image):
    # extract a random crop from the image with the target width and height
    return extract_patches_2d(image, (self.width, self.height), max_patches=1)[0]
