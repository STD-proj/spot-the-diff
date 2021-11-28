# Imports:
import argparse
import os
import sys
from builtins import filter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from IP.feature_points import *
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, isdir, getsize
from os import makedirs
from keras.preprocessing.image import ImageDataGenerator
import glob
import shutil
from DL.DeepLearningFunctions import DeepLearning
from PIL import Image

# Consts:
img_height = 200
img_width = 200

# Help functions:
'''
This function converts image's format from any type into jpeg type.
'''
def convertJPG200(imagelist, path, toformat, size=(100, 100), keyword=''):
    print("Start convert items that contain '{}'.".format(keyword))
    _matching_image = [_image for _image in imagelist if keyword in _image]  # create a list of images to convert
    _index = 150 - len(_matching_image)  # define first index

    for _image in _matching_image:
        _img = Image.open(path + _image).convert('RGB')  # convert to RGB to be able to save in other format
        _new_img = _img.resize(size)  # resize image into specific size
        _ext = _image[-3:]
        _format = 'JPEG' if _ext.lower() == 'jpg' else _ext.upper()
        _new_img.save(path + str(_index) + toformat, _format)  # save image in jpg format
        _index += 1

    print("Finished convert {} items.\n".format(len(_matching_image)))

'''
This function removes all images from a directory which named within a given keyword. 
'''
def removeImageFromDir(imagelist, path, keyword=''):
    print("Start removing items that contain '{}'.".format(keyword))
    for _image in imagelist:
        if keyword in _image:  # if keyword in image's name
            os.remove(path + os.sep + _image)  # remove image from directory
    print("Finished removing.\n")

'''
This function calculates correct prediction for the algorithm.
'''
def correctPrediction(test_generator, y_pred, algorithm):
    correct = 0
    for i, f in enumerate(test_generator.filenames):
        if f.startswith('bad_changes') and y_pred[i] < 0.5:
            correct += 1
        if f.startswith('good_changes') and y_pred[i] >= 0.5:
            correct += 1

    print('Correct predictions for {}: '.format(algorithm) + str(round(100 * correct / len(test_generator.filenames), 2)) + "%")

'''
This function draws 6 test images with a graph of probability of good changes. 
'''
def draw(method, label_dict, batch_size, x_test, y_test, p, size=(12, 12)):

    fig = plt.figure(figsize=size)
    fig.canvas.set_window_title(method)

    for i in range(batch_size):
        plt.subplot(batch_size, 2, 2 * i + 1)
        plt.imshow(x_test[i])
        plt.title(label_dict[y_test[i]])

        plt.subplot(batch_size, 2, 2 * i + 2)
        plt.bar(range(2), p[i])
        plt.xticks(range(2), ['bad changes', 'good changes'])

    plt.show()

def clear_dir(directory):
    if isdir(directory):  # if directory already exists
        shutil.rmtree(directory)

def copy_images_into_folder(folder, set_data, labels_data, dir1, dir2):
    for f, i in zip(set_data, labels_data):
        if i == 0:
            shutil.copy2(f, folder + '/{}/'.format(dir1))
        else:
            shutil.copy2(f, folder + '/{}/'.format(dir2))


def keras_resnet_model():
    # Definitions:
    dirpath = '../Data/'
    dir_good_changes = 'good_changes/'
    dir_bad_changes = 'bad_changes/'
    formatJPG = '.jpg'
    dirnet = 'net/'
    imagesize = (200, 200)

    # create datasets in a standard format (name, size) and remove previous items
    # for bad changes:
    path_bad_changes = dirpath + dir_bad_changes  # define path
    list_bad_changes = os.listdir(path_bad_changes)  # get files' names in path into a list
    convertJPG200(list_bad_changes, path_bad_changes, formatJPG, size=imagesize,
                  keyword='bad_changes')  # convert all files in the above list that named with keyword
    removeImageFromDir(list_bad_changes, path_bad_changes, 'bad_changes')  # remove previous data from path

    # for good changes:
    path_good_changes = dirpath + dir_good_changes  # define path
    list_good_changes = os.listdir(path_good_changes)  # get files' names in path into a list
    convertJPG200(list_good_changes, path_good_changes, formatJPG, size=imagesize,
                  keyword='good_changes')  # convert all files in the above list that named with keyword
    removeImageFromDir(list_good_changes, path_good_changes, 'good_changes')  # remove previous data from path

    files_good = glob.glob(path_good_changes + '/*' + formatJPG)
    files_bad = glob.glob(path_bad_changes + '/*' + formatJPG)

    files = files_good + files_bad
    half_files = int(len(files) / 2 + 1)
    labels = np.array([0] * 150 + [1] * 150)  # for later calculations

    # for later calculations
    size = np.zeros(len(files))
    for i, f in enumerate(files):
        size[i] = getsize(f)
    idx = np.where(size == 0)[0]
    for i in idx[::-1]:
        del (files[i])
        labels = np.delete(labels, i)

    len_data = len(files)
    train_examples = 130  # define amount of files to be used as train examples
    test_examples = len_data - train_examples  # calculate amount of files to be used as test examples

    # randomly choose <len_data> files as training and testing cases
    permutation = np.random.permutation(len_data)
    train_set = [files[i] for i in permutation[:train_examples]]
    test_set = [files[i] for i in permutation[-test_examples:]]
    train_labels = labels[permutation[:train_examples]]
    test_labels = labels[permutation[-test_examples:]]

    train_folder = dirpath + dirnet + 'train'
    test_folder = dirpath + dirnet + 'test'

    clear_dir(train_folder)  # clear directory if it already exists
    clear_dir(test_folder)  # clear directory if it already exists

    # create folders for good changes and bad changes in train and test folders
    makedirs(train_folder + '/{}/'.format(dir_bad_changes))
    makedirs(train_folder + '/{}/'.format(dir_good_changes))
    makedirs(test_folder + '/{}/'.format(dir_bad_changes))
    makedirs(test_folder + '/{}/'.format(dir_good_changes))

    # copy images from source directories into new directories that where created
    copy_images_into_folder(train_folder, train_set, train_labels, dir_bad_changes, dir_good_changes)
    copy_images_into_folder(test_folder, test_set, test_labels, dir_bad_changes, dir_good_changes)

    # generate data
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)

    label_dict = {0: 'bad_changes', 1: 'good_changes', 2: 'New change'}

    # start Keras algorithm and show results on a graph
    print("Start Keras")
    keras_model = DeepLearning()
    keras_model.Keras()

    # start Resnet on Keras algorithm and show results on a graph
    print("Start Resnet on Keras")
    resnet_model = DeepLearning()
    resnet_model.Resnet()

    # generate train data
    batch_size = 128
    train_generator = datagen.flow_from_directory(
        train_folder,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # iteration of the training data in batches
    keras_model.fit_generator(train_generator, train_examples, batch_size, epoch=10)
    resnet_model.fit_generator(train_generator, train_examples, batch_size, epoch=10)

    # generate test data
    batch_size = 1
    test_generator = datagen.flow_from_directory(
        test_folder,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    # generate predictions on new data
    y_pred_keras = keras_model.predict_generator(test_generator, test_examples, batch_size, worker=4)
    y_pred_resnet = resnet_model.predict_generator(test_generator, test_examples, batch_size, worker=4)

    correctPrediction(test_generator, y_pred_keras, 'Keras')
    correctPrediction(test_generator, y_pred_resnet, 'Resnet')

    return datagen, keras_model, y_pred_keras, label_dict, resnet_model, y_pred_resnet


# Program:
def main(image):
    delete_all_files_in_folder()  # delete all previous changed images

    # build basic model of Keras and Resnet
    datagen, keras_model, y_pred_keras, label_dict, resnet_model, y_pred_resnet = keras_resnet_model()

    fp_controller = FeaturePointsController(image)
    check_folder = f"{DIRECTORY}/test"
    fp_controller.run(1, False, check_folder)  # run a function to create a change in an image

    batch_size = 1
    test_generator = datagen.flow_from_directory(
        DIRECTORY,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)
    x_test, y_test = next(test_generator)
    y_test = [2]

    p = keras_model.predict(x_test)  # generate predictions on new change image
    p = np.hstack([y_pred_keras, 1 - y_pred_keras])
    method = "Keras"
    draw(method, label_dict, batch_size, x_test, y_test, p, size=(12, 12))
    print("End Keras\n")

    p = resnet_model.predict(x_test)  # generate predictions on new data
    resnet_model.model.save('resnet_model.h5')
    p = np.hstack([y_pred_resnet, 1 - y_pred_resnet])
    method = "Resnet"
    draw(method, label_dict, batch_size, x_test, y_test, p, size=(12, 12))
    print("End Resnet on Keras")

    # show predictions on graph
    plt.plot(y_pred_keras, label="Keras")
    plt.plot(y_pred_resnet, label="Resnet")
    plt.title("Predictions")
    plt.xlabel("Pic index")
    plt.ylabel("Prediction")
    plt.legend()


if __name__ == '__main__':
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True,
                        help="path to input image")
        args = vars(ap.parse_args())
        image = args["image"]
        main(image)  # run program
        plt.show()  # show linear regression on graph
    except Exception as e:
        print(f'Can not run program due to the next reason: {e}')
        exit()
