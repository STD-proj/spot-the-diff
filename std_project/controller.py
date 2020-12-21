
import argparse
from object_detection import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
   help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
   help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
   help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
   help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

yolo_object_detector = YoloObjectDetector(args["image"], args["yolo"],
                                args["confidence"], args["threshold"])
data = yolo_object_detector.detect_objects()