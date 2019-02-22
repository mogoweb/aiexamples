import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

import sys
sys.path.append("../..")

from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.patch_preprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.mean_preprocessor import MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io.hdf5_dataset_generator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.alexnet import AlexNet


aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug, preprocessors=[pp, mp, iap], classes=2)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, 128, preprocessors=[sp, mp, iap], classes=2)

print("[INFO] compiling model ...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

model.fit_generator(train_gen.generator(), steps_per_epoch=train_gen.num_images,
                    validation_data=val_gen.generator(), validation_steps=val_gen.num_images,
                    epochs=75, max_queue_size=128 * 2, callbacks=callbacks, verbose=1)

print("[INFO] serializing model ...")
model.save(config.MODEL_PATH, overwrite=True)

train_gen.close()
val_gen.close()
