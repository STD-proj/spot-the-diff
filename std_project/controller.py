
import argparse
from feature_points import *
from object_detection import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
   help="path to input image")
args = vars(ap.parse_args())

# yolo_object_detector = YoloObjectDetector(args["image"], args["yolo"],
#                                 args["confidence"], args["threshold"])
# data = yolo_object_detector.detect_objects()

fp_controller = FeaturePointsController(args["image"])
fp_controller.run()