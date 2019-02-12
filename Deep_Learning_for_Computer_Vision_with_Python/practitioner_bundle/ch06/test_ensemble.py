from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True, help="path to models directory")
args = vars(ap.parse_args())


(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "trunk"]
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

model_paths = os.path.sep.join(args["models"], "*.model")
model_paths = list(glob.glob(model_paths))
models = []

for (i, model_path) in enumerate(model_paths):
  print("[INFO] loading model {}/{}".format(i + 1, len(model_paths)))
  models.append(load_model(model_path))


print("[INFO] evaluating ensemble ...")
predictions = []
for model in models:
  predictions.append(model.predict(testX, batch_size=64))

predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
