# define the path to the image directory
IMAGES_PATH = "../../datasets/kaggle_dogs_vs_cats/train"

NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define the path to the output training, validation and testing HDF5 files
TRAIN_HDF5 = "../../datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
VAL_HDF5 = "../../datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = "../../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = "output/alexnet_dogs_vs_cats.model"

# define the path to the dataset mean
DATASET_MEAN = "output/dogs_vs_cats_mean.json"

# define the path to the output directory used for storing plots, classification reports, etc
OUTPUT_PATH = "output"