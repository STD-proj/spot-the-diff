import cv2
from scipy import ndimage
from PIL import Image as im
from pathlib import Path
import os
import glob
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pathlib
import pandas as pd
import shutil

# Consts:
AMOUNT_OF_RUNS = 5
IMG_SUFFIX = ".jpg"
DIRECTORY = "Changes"

def read_img(img_path):
    '''
    This function reads an image from a given path
    :param img_path: a path of an image
    :return: the read image
    '''
    return cv2.imread(img_path)

def show_img(img, title):
    '''
    This image shows an image
    :param img: an image
    :param title: a title of the image
    :return: None
    '''
    cv2.imshow(title, img)

def save_img(img, name, directory, index=0, crop=(0, 0, 0, 0)):
    '''
    This function saves an image.
    :param img: an image
    :param name: a name for the image
    :param index: index of image, default sets as 0
    :param crop: a crop in the given image
    :return: None
    '''

    if not os.path.isdir(directory):
        os.mkdir(directory)

    crop_name = f"_{crop[0]}_{crop[1]}_{crop[2]}_{crop[3]}"
    cv2.imwrite(os.path.join(directory, name + '_' + str(index) + crop_name + IMG_SUFFIX), img)

# -- General functions --
def delete_all_files_in_folder(folder=DIRECTORY):
    '''
    This function deletes all files in a given folder.
    :param folder: The folder to delete from. The default directory is the constant DIRECTORY.
    :return: None
    '''
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_changed_image(directory):
    '''
    This function extracts the changed image.
    :return: the changed image
    '''
    return glob.glob(f"{directory}/*{IMG_SUFFIX}")[0]
