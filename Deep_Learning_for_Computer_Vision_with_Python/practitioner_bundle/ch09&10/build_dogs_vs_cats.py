from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import progressbar
import os
import json
import cv2

import sys
sys.path.append("../..")
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.io.hdf5_dataset_writer import HDF5DatasetWriter


# grab the paths to the images
train_paths = list(paths.list_images(config.IMAGES_PATH))
train_labels = [p.split(os.path.sep)[-1].split(".")[0] for p in train_paths]

le = LabelEncoder()
train_labels = le.fit_transform(train_labels)


split = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels,
                         random_state=42)
(train_paths, test_paths, train_labels, test_labels) = split

split = train_test_split(train_paths, train_labels, test_size=config.NUM_VAL_IMAGES, stratify=train_labels,
                         random_state=42)
(train_paths, val_paths, train_labels, val_labels) = split

# construct a list pairing the training, validation, and testing images paths along with their corresponding
# labels and output HDF5 files
datasets = [
  ("train", train_paths, train_labels, config.TRAIN_HDF5),
  ("val", val_paths, val_labels, config.VAL_HDF5),
  ("test", test_paths, test_labels, config.TEST_HDF5)
]

aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

for (dtype, paths, labels, output_path) in datasets:
  # create HDF5 writer
  print("(INFO) building {} ...".format(output_path))
  writer = HDF5DatasetWriter((len(paths), 256, 256, 3), output_path)

  # initialize the progress bar
  widgets = ["Building dataset:", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
  pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

  # loop over the image paths
  for (i, (path, label)) in enumerate(zip(paths, labels)):
    image = cv2.imread(path)
    image = aap.preprocess(image)

    if dtype == "train":
      (b, g, r) = cv2.mean(image)[:3]
      R.append(r)
      G.append(g)
      B.append(b)

    writer.add([image], [label])
    pbar.update(i)

  pbar.finish()
  writer.close()

# construct a dictionary of averages, then serialize the means to a JSON file
print("[INFO] serialize means ...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()