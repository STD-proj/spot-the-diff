import cv2
from scipy import ndimage
from PIL import Image as im
# from skimage import transform
# from skimage import io
from pathlib import Path
import os
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pathlib

END_PIC = ".jpg"
DIRECTORY = "Changes"

if not os.path.isdir(DIRECTORY):
    os.mkdir(DIRECTORY)

def read_img(img_path):
    return cv2.imread(img_path)

def show_img(img, title):
    cv2.imshow(title, img)

def save_img(img, name, index=0):
    cv2.imwrite(os.path.join(DIRECTORY, name + '_' + str(index) + END_PIC), img)
