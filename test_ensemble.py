# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True, help="path to models directory")
args =vars(ap.parse_args())

# load the testing data, then scale it into the range [0, 1]
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") /255.0

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

# convert the labels from integers to vectors
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# construct the path used to collect the models then initialize the  models list
modelPaths = os.path.sep.join([args["models"], "*.model"])
modelPaths =list(glob.glob(modelPaths))
models = []