import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import *
from keras.applications.resnet import ResNet50
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adagrad
from keras.applications.vgg16 import VGG16

import numpy as np
import cv2

dim = (150, 150)

def create_data():
    X_image_train1 = []
    X_image_train2 = []
    y_train = []

    X_image_test1 = []
    X_image_test2 = []
    y_test = []

    inside_folders = ['good_changes', 'bad_changes']

    path_org = "../Data/examples/"
    path_new_base = "../Data"

    num_train, num_test = 0, 0
    for folder in inside_folders:
        num_train += len(os.listdir(f"{path_new_base}/net/train/{folder}/"))
        num_test += len(os.listdir(f"{path_new_base}/net/test/{folder}/"))

    for folder in inside_folders:
        path_new = f"{path_new_base}/net/train/{folder}/"
        files_new = os.listdir(path_new)
        files_org = [x.split("_")[0] + '.jpg' for x in files_new]

        for i in range(len(files_new)):
            img_org = cv2.imread(path_org + files_org[i])
            img_org = cv2.resize(img_org, dim, interpolation=cv2.INTER_AREA)
            img_new = cv2.imread(path_new + files_new[i])
            img_new = cv2.resize(img_new, dim, interpolation=cv2.INTER_AREA)
            X_image_train1.append(img_org)
            X_image_train2.append(img_new)
            y_train.append(int(folder == 'good_changes'))

    for folder in inside_folders:
        path_new = f"{path_new_base}/net/test/{folder}/"
        files_new = os.listdir(path_new)
        files_org = [x.split("_")[0] + '.jpg' for x in files_new]

        for i in range(len(files_new)):
            img_org = cv2.imread(path_org + files_org[i])
            img_org = cv2.resize(img_org, dim, interpolation=cv2.INTER_AREA)
            img_new = cv2.imread(path_new + files_new[i])
            img_new = cv2.resize(img_new, dim, interpolation=cv2.INTER_AREA)
            X_image_test1.append(img_org)
            X_image_test2.append(img_new)
            y_test.append(int(folder == 'good_changes'))

    return X_image_train1, X_image_train2, y_train, X_image_test1, X_image_test2, y_test

def create_model():
    image_input = Input(shape=(dim[0], dim[1], 3), name='image1')
    vgg16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(dim[0], dim[1], 3))(image_input)
    x = Flatten()(vgg16)
    x = Dense(2, activation='relu')(x)

    image_input2 = Input(shape=(dim[0], dim[1], 3), name='image2')
    resnet = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(dim[0], dim[1], 3))(image_input2)
    y = Flatten()(resnet)
    y = Dense(2, activation='relu')(y)

    concatenated = concatenate([x, y], axis=-1)
    output = Dense(1, activation='sigmoid')(concatenated)
    model = Model([image_input, image_input2], output)
    model.summary()
    return model



X_image_train1, X_image_train2, y_train, X_image_test1, X_image_test2, y_test = create_data()
X_image_train1 = np.array(X_image_train1)
X_image_train2 = np.array(X_image_train2)
y_train       = np.array(y_train)
perm = np.arange(y_train.shape[0])
np.random.shuffle(perm)
X_image_train1 = X_image_train1[perm]
X_image_train2   = X_image_train2[perm]
y_train       = y_train[perm]

X_image_test1 = np.array(X_image_test1)
X_image_test2 = np.array(X_image_test2)
y_test       = np.array(y_test)
perm = np.arange(y_test.shape[0])
np.random.shuffle(perm)
X_image_test1 = X_image_test1[perm]
X_image_test2   = X_image_test2[perm]
y_test       = y_test[perm]

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit([X_image_train1, X_image_train2], y_train,
                    # validation_data=([X_image_test1, X_image_test2], y_test),
                    epochs=5,
                    batch_size=16)
# serialize the model to disk
print("[INFO] saving siamese model...")
model.save('./siamese_mode.h5')
pred = model.predict([X_image_test1, X_image_test2])
print(pred)