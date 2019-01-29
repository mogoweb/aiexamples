from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os
import sys
sys.path.append("../..")
from pyimagesearch.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.fcheadnet import FCHeadNet


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


print("[INFO] loading images ...")
image_paths = list(paths.list_images(args["dataset"]))
class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# load the VGG16 network, ensure the head FC layer sets are left off
base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# initialize the new head of the network, a set of FC layers followed by a softmax classifier
head_model = FCHeadNet.build(base_model, len(class_names), 256)

# place the head FC model on top of the base model -- this will become the actual model we will train
model = Model(inputs=base_model.input, outputs=head_model)

# loop over all layers in the base model and freeze them so they will not be updated during the training process
for layer in base_model.layers:
  layer.trainable = False

print("[INFO] compiling model ...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network for a few epochs -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head ...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // 32, epochs=25, verbose=1)

print("[INFO] evaluating after initialization ...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=class_names))

# now that the head FC layers have been initialized/trained, let's unfreeze CONV layers and make then trainable
for layer in base_model.layers[15:]:
  layer.trainable = True

print("[INFO] re-compiling model ...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network for a few epochs -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] fine-tuning head ...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // 32, epochs=25, verbose=1)

print("[INFO] evaluating after fine-tuning ...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=class_names))

print("[INFO] serializing model ...")
model.save(args["model"])