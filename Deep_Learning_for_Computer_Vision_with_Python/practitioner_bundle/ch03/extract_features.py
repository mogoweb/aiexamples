from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os
import sys

sys.path.append("../..")
from pyimagesearch.io.hdf5_dataset_writer import HDF5DatasetWriter


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output HDF5 file")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer_size", type=int, default=100, help="size of feature extraction buffer")
args = vars(ap.parse_args())


# batch size
bs = args["batch_size"]

print("[INFO] loading images ...")
image_paths = list(paths.list_images(args["dataset"]))
random.shuffle(image_paths)

labels = [p.split(os.path.sep)[-2] for p in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading network ...")
model = VGG16(weights="imagenet", include_top=False)

dataset = HDF5DatasetWriter((len(image_paths), 512 * 7 * 7), args["output"], data_key="features", buf_size=args["buffer_size"])
dataset.store_class_labels(le.classes_)

widget = ["Extracting features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widget).start()

for i in np.arange(0, len(image_paths), bs):
  batch_paths = image_paths[i : i+bs]
  batch_labels = labels[i : i + bs]
  batch_images = []

  for (j, image_path) in enumerate(batch_paths):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    batch_images.append(image)

  batch_images = np.vstack(batch_images)
  features = model.predict(batch_images, batch_size=bs)
  features = features.reshape((features.shape[0], 512 * 7 * 7))
  print(features.shape)
  dataset.add(features, batch_labels)
  pbar.update(i)

dataset.close()
pbar.finish()