from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import numpy as np
import argparse

import glob
import matplotlib.pyplot as plt
from PIL import Image
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-o", "--output", required=True, help="path to output directory to store augmentation examples")
ap.add_argument("-p", "--prefix", type=str, default="image", help="output filename prefix")
args = vars(ap.parse_args())

# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image ...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# construct the image generator for data augmentation then
# initialize the total number of images generated thus far
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

aug.fit(image)
# construct the actual Python generator
print("[INFO] generating images ...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"], save_prefix=args["prefix"],
                    save_format="jpeg")

total = 0
# loop over examples from out image data augmentation generator
for image in imageGen:
  # increment out counter
  total += 1

  if total == 10:
    break

# 找到本地生成图，把10张图打印到同一张figure上
name_list = glob.glob(os.path.join(args["output"], "*"))
fig = plt.figure()
for i in range(10):
    img = Image.open(name_list[i])
    sub_img = fig.add_subplot(2, 5, i+1)
    sub_img.imshow(img)
plt.show()