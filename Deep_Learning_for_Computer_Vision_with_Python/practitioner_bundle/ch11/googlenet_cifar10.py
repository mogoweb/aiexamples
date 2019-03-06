import matplotlib
matplotlib.use("Agg")


from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os
import sys

sys.path.append("../..")
from pyimagesearch.nn.conv.minigooglenet import MiniGoogleNet
# from pyimagesearch.callbacks.training_monitor import TrainingMonitor


NUM_EPOCHS = 70
INIT_LR = 5e-3


def poly_decay(epoch):
  max_epochs = NUM_EPOCHS
  base_lr = INIT_LR
  power = 1.0

  alpha = base_lr * (1 - (epoch / float(max_epochs))) ** power

  return alpha


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data ...")
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
train_x = train_x.astype("float")
test_x = test_x.astype("float")

mean = np.mean(train_x, axis=0)
train_x -= mean
test_x -= mean

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
# callbacks = [TrainingMonitor(fig_path, json_path=json_path), LearningRateScheduler(poly_decay)]
callbacks = [LearningRateScheduler(poly_decay)]

print("[INFO] compiling model ...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogleNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training model ...")
model.fit_generator(aug.flow(train_x, train_y, batch_size=64), validation_data=(test_x, test_y),
                    steps_per_epoch=len(train_x) // 64, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

print("[INFO] serializing network ...")
model.save(args["model"])
