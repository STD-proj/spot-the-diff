# Imports:
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten,\
    BatchNormalization, MaxPooling2D
from keras import layers
from keras import models

# Class
class DeepLearning:
    def __init__(self, input_shape=(200, 200, 3), img_height=200, img_width=200, channels=3, sequential=True):
        self.input_shape = input_shape
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels

        if sequential:
            # start implementation of Keras
            self.model = Sequential()  # the simplest type of model is the Sequential model, a linear stack of layers
        else:
            self.model = None

    # Functions to help with calculations in main
    def fit_generator(self, train_generator, train_examples, batch_size, epoch=2):
        return self.model.fit_generator(train_generator, train_examples // batch_size, epochs=epoch)

    def predict_generator(self, test_generator, test_examples, batch_size, worker=4):
        return self.model.predict_generator(test_generator, test_examples // batch_size, workers=worker)

    def predict(self, x_test):
        return self.model.predict(x_test)

    # Function to help with calculation of Resnet algorithm
    def residual_network(self, x):
        def add_common_layers(y):
            y = layers.BatchNormalization()(y)
            y = layers.LeakyReLU()(y)

            return y

        def grouped_convolution(y, nb_channels, _strides):
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

        def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
            """
            The network consists of a stack of residual blocks. These blocks have the same topology,
            and are subject to two simple rules:
            - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
            - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
            """
            shortcut = y

            # modify the residual building block as a bottleneck design to make the network more economical
            y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
            y = add_common_layers(y)

            # ResNet
            y = grouped_convolution(y, nb_channels_in, _strides=_strides)
            y = add_common_layers(y)

            y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
            # batch normalization is employed after aggregating the transformations and before adding to the shortcut
            y = layers.BatchNormalization()(y)

            # identity shortcuts used directly when the input and output are of the same dimensions
            if _project_shortcut or _strides != (1, 1):
                # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(
                    shortcut)
                shortcut = layers.BatchNormalization()(shortcut)

            y = layers.add([shortcut, y])

            # relu is performed right after each batch normalization,
            # expect for the output of the block where relu is performed after the adding to the shortcut
            y = layers.LeakyReLU()(y)

            return y

        # conv1
        x = layers.Conv2D(2, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        x = add_common_layers(x)

        # conv2
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        for i in range(3):
            project_shortcut = True if i == 0 else False
            x = residual_block(x, 4, 8, _project_shortcut=project_shortcut)

        # conv3
        for i in range(4):
            # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 8, 16, _strides=strides)

        # conv4
        for i in range(6):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 16, 32, _strides=strides)

        # conv5
        for i in range(3):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 32, 64, _strides=strides)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1)(x)

        return x

    def Keras(self):
        # stacking layers with .add()
        self.model.add(Conv2D(8, kernel_size=(3, 3), padding='same', input_shape=(self.img_width, self.img_height, self.channels)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPool2D(pool_size=(3, 3)))

        self.model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))

        # configure the learning process with .compile()
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def Resnet(self):
        image_tensor = layers.Input(shape=(self.img_height, self.img_width,self.channels))
        print(image_tensor)

        network_output = self.residual_network(image_tensor)

        self.model = models.Model(inputs=[image_tensor], outputs=[network_output])
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
