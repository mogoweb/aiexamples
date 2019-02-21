from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
  def __init__(self, db_path, batch_size, preprocessors=None, aug=None, binarize=True, classes=2):

    self.batch_size = batch_size
    self.preprocessors = preprocessors
    self.aug = aug
    self.binarize = binarize
    self.classes = classes

    self.db = h5py.File(db_path)
    self.num_images = self.db["labels"].shape[0]


  def generator(self, passes=np.inf):
    epochs = 0

    # keep looping infinitely -- the model will stop once we have reach the desired number of epochs
    while epochs < passes:
      # loop over the HDF5 dataset
      for i in np.arange(0, self.num_images, self.batch_size):
        # extract the images and labels from HDF5 dataset
        images = self.db["images"][i : i + self.batch_size]
        labels = self.db["labels"][i : i + self.batch_size]

        if self.binarize:
          labels = np_utils.to_categorical(labels, self.classes)

        if self.preprocessors is not None:
          proc_images = []
          for image in images:
            for p in self.preprocessors:
              image = p.preprocess(image)

            proc_images.append(image)

          images = np.array(proc_images)

        if self.aug is not None:
          (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batch_size))
          yield(images, labels)

      epochs += 1


  def close(self):
    self.db.close()