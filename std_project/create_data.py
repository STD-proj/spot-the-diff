
import argparse
from IP.feature_points import *
from IP.object_detection import *
import glob

PATH = "examples/"
DIR = "Data/all/"

# Create changes for each image exists in a given directory
for image in glob.glob(f"{PATH}*{IMG_SUFFIX}"):
    fp_controller = FeaturePointsController(image)
    fp_controller.run(None, True, DIR)



