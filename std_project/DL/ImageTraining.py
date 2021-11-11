# Imports:
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

    print('Correct predictions for {}: '.format(algorithm) + str(correct / len(test_generator.filenames)))

'''
This function draws 6 test images with a graph of probability of good changes. 
'''
def draw(label_dict, batch_size, x_test, y_test, p, size=(12, 12)):

    plt.figure(figsize=size)

    for i in range(batch_size):
        plt.subplot(batch_size, 2, 2 * i + 1)
        plt.imshow(x_test[i])
        plt.title(label_dict[y_test[i]])

        plt.subplot(batch_size, 2, 2 * i + 2)
        plt.bar(range(2), p[i])
        plt.xticks(range(2), ['bad changes', 'good changes'])

    plt.show()

def delete_directory(folder):
    if isdir(folder):  # if directory already exists
        shutil.rmtree(folder)

def create_directory(folder, dir_bad_changes, dir_good_changes):
    makedirs(folder + '/{}/'.format(dir_bad_changes))
    makedirs(folder + '/{}/'.format(dir_good_changes))


# Program:
def main():
    # Definitions:
    dirpath = '../Data/'
    dir_good_changes = 'good_changes/'
    dir_bad_changes = 'bad_changes/'
    formatJPG = '.jpg'
    dirnet = 'net/'
    source_dir = '../examples/'
    imagesize = (200, 200)

    # create datasets in a standard format (name, size) and remove previous items
    # for others:
    pathother = dirpath + dir_bad_changes  # define path
    listother = os.listdir(pathother)  # get files' names in path into a list
    convertJPG200(listother, pathother, formatJPG, size=imagesize, keyword='bad_changes')  # convert all files in the above list that named with keyword
    removeImageFromDir(listother, pathother, 'bad_changes')  # remove previous data from path

    # for good changes:
    path_good_changes = dirpath + dir_good_changes  # define path
    list_good_changes = os.listdir(path_good_changes)  # get files' names in path into a list
    convertJPG200(list_good_changes, path_good_changes, formatJPG, size=imagesize, keyword='good_changes')  # convert all files in the above list that named with keyword
    removeImageFromDir(list_good_changes, path_good_changes, 'good_changes')  # remove previous data from path

    # files = glob.glob(dirpath + '/**/*' + formatJPG)  # get files' names that end with .jpg
    files = glob.glob(dirpath + dir_good_changes + '/*' + formatJPG)  # get files' names that end with .jpg
    files += glob.glob(dirpath + dir_bad_changes + '/*' + formatJPG)  # get files' names that end with .jpg
    source_files = glob.glob(source_dir + '/*' + formatJPG)  # get files' names that end with .jpg
    labels = np.array([0]*150 + [1]*150)  # for later calculations

    # for later calculations
    size = np.zeros(len(files))
    for i, f in enumerate(files):
        size[i] = getsize(f)

    idx = np.where(size == 0)[0]
    for i in idx[::-1]:
        del(files[i])
        labels = np.delete(labels, i)

    len_data = len(files)
    train_examples = len_data  # define amount of files to be used as train examples
    test_examples = len(source_files)  # calculate amount of files to be used as test examples

    # randomly choose files as training and testing cases
    permutation = np.random.permutation(len_data)
    train_set = [files[i] for i in permutation[:train_examples]]
    test_set = [source_files[i] for i in range(test_examples)]
    train_labels = labels[permutation[:train_examples]]
    test_labels = labels[permutation[-test_examples:]]

    train_folder = dirpath + dirnet + 'train'
    test_folder = dirpath + dirnet + 'test'

    delete_directory(train_folder)
    delete_directory(test_folder)

    # create folders for bad and good changes in train and test folders
    create_directory(train_folder, dir_bad_changes, dir_good_changes)
    create_directory(test_folder, dir_bad_changes, dir_good_changes)

    # copy images from source directories into new directories that where created
    for f, i in zip(train_set, train_labels):
        if i == 0:
            shutil.copy2(f, train_folder + '/{}/'.format(dir_bad_changes))
        else:
            shutil.copy2(f, train_folder + '/{}/'.format(dir_good_changes))

    for f, i in zip(test_set, test_labels):
        if i == 0:
            shutil.copy2(f, test_folder + '/{}/'.format(dir_bad_changes))
        else:
            shutil.copy2(f, test_folder + '/{}/'.format(dir_good_changes))

    # generate data
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)

    img_height = 200
    img_width = 200
    label_dict = {0: 'bad_changes', 1: 'good_changes'}

    # start Keras algorithm and show results on a graph
    print("Start Keras")
    keras_model = DeepLearning()
    keras_model.Keras()

    # start VGG16 on Keras algorithm and show results on a graph
    print("Start VGG16 on Keras")
    vgg16_model = DeepLearning()
    vgg16_model.VGG16()

    # start Resnet on Keras algorithm and show results on a graph
    print("Start Resnet on Keras")
    resnet_model = DeepLearning()
    resnet_model.Resnet()

    # generate train data
    batch_size = train_examples
    train_generator = datagen.flow_from_directory(
        train_folder,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # iteration of the training data in batches
    keras_model.fit_generator(train_generator, train_examples, batch_size, epoch=2)
    vgg16_model.fit_generator(train_generator, train_examples, batch_size, epoch=2)
    resnet_model.fit_generator(train_generator, train_examples, batch_size, epoch=2)

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
    y_pred_vgg16 = vgg16_model.predict_generator(test_generator, test_examples, batch_size, worker=4)
    y_pred_resnet = resnet_model.predict_generator(test_generator, test_examples, batch_size, worker=4)

    correctPrediction(test_generator, y_pred_keras, 'Keras')
    correctPrediction(test_generator, y_pred_vgg16, 'VGG16')
    correctPrediction(test_generator, y_pred_resnet, 'Resnet')

    batch_size = 6
    test_generator = datagen.flow_from_directory(
        test_folder,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)
    x_test, y_test = next(test_generator)

    p = keras_model.predict(x_test)  # generate predictions on new data
    p = np.hstack([y_pred_keras, 1 - y_pred_keras])
    draw(label_dict, batch_size, x_test, y_test, p, size=(12, 12))
    print("End Keras\n")

    p = vgg16_model.predict(x_test)  # generate predictions on new data
    p = np.hstack([y_pred_vgg16, 1 - y_pred_vgg16])
    draw(label_dict, batch_size, x_test, y_test, p, size=(12, 12))
    print("End VGG16 on Keras\n")

    p = resnet_model.predict(x_test)  # generate predictions on new data
    p = np.hstack([y_pred_resnet, 1 - y_pred_resnet])
    draw(label_dict, batch_size, x_test, y_test, p, size=(12, 12))
    print("End Resnet on Keras")


if __name__ == '__main__':
    main()
