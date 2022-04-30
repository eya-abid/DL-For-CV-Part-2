# import the necessary packages
from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import oss

# grab the paths to the images
trainPaths =list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[2].split(".")[0] for p in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling from the training set to build the testing split from the training data
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# perform another stratified sampling, this time to build the validation data
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

# construct a list pairing the training, validation, and testing image paths along with their corresponding labels and output HDF5 files
datasets = [
("train", trainPaths, trainLabels, config.TRAIN_HDF5),
("val", valPaths, valLabels, config.VAL_HDF5),
("test", testPaths, testLabels, config.TEST_HDF5)]

# initialize the image preprocessor and the lists of RGB channel averages
aap = AspectAwarePreprocessor(256,256)
(R, G, B) = ([], [], [])