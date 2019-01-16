
import os
import glob
import datetime
import argparse
import shutil

# print start time
print("[INFO] program started on - " + str(datetime.datetime.now))

ap = argparse.ArgumentParser()
ap.add_argument("-j", "--jpg", required=True, help="path to 17 flowers jpgs(all in one folder)")
ap.add_argument("-o", "--output", required=True, help="path to organized 17 flowers")
args = vars(ap.parse_args())

# we only have 17 categories
class_limit = 17

# take all the images from the dataset
image_paths = sorted(glob.glob(os.path.join(args["jpg"], "*.jpg")), key=os.path.basename)

# variables to keep track
i = 0

# flower17 class names
class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
               "iris", "tigerlily", "tulip", "fritillary", "sunflower",
               "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
               "windflower", "pansy"]

# loop over the class labels
for x in range(0, class_limit):
  # create a folder for that class
  cur_path = os.path.join(args["output"], class_names[x])
  os.makedirs(cur_path, exist_ok=True)

  # loop over the images in the dataset
  for image_path in image_paths[i:i+80]:
    original_path = image_path
    image_path = os.path.basename(image_path)
    shutil.copyfile(original_path, os.path.join(cur_path, image_path))

  i += 80

# print end time
print("[INFO] program ended on - " + str(datetime.datetime.now))