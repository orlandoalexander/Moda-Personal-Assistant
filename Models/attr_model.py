import numpy as np
import pandas as pd

from keras.layers import Input, Dense, Flatten
from keras.metrics import Precision, Recall

from keras.applications import InceptionV3
from keras.applications import inception_v3
from keras.applications import ResNet50
from keras.applications import resnet50
from keras.applications import MobileNetV2
from keras.applications import mobilenet_v2
from keras.applications import EfficientNetB0
from keras.applications import efficientnet

# attribute should be one of the following strings:
# 'design', 'sleeves', 'length', 'neckline', 'fabric', 'fit'

# model should be one of the following strings:
# 'inception', 'resnet', 'mobilenet', 'efficientnet'

class AttrModel:
    def __init__(self, attribute, model, input_shape, final_layer_neurons,
                 X_train, X_test,
                 y_train, y_test, **kwargs):
        self.attribute = attribute
        self.model = model.lower()
        self.input_shape = input_shape
        self.final_layer_neurons = final_layer_neurons
        self.kwargs = kwargs
        self.data = self.get_data()
        self.X_train = X_train
        self.X_test = X_test
        self. y_train = y_train
        self.y_test = y_test

        self.cat_nums = {
            'design': 7,
            'sleeves': 3,
            'length': 2,
            'neckline': 4,
            'fabric': 6,
            'fit': 3
        }

        self.cat_num = self.cat_nums[self.attribute]
        self.model = self.instantiate_model()  # calling the function below
        self.activation = 'sigmoid' if self.attribute == 'length' else 'softmax'
        self.loss = 'categorical_crossentropy' if self.attribute == 'length' else 'binary_crossentropy'

    def instantiate_model(self):
        input_tensor = Input(shape=self.input_shape) # input_shape is a tuple passed to the class
        if self.model == 'inception':              # calling the chosen pretrained model
            base_model = InceptionV3(include_top=False, weights='imagenet',
                                     classes=self.cat_num, input_tensor=input_tensor)
        elif self.model == 'resnet':
            base_model = ResNet50(include_top=False, weights='imagenet',
                                  classes=self.cat_num, input_tensor=input_tensor)
        elif self.model == 'mobilenet':
            base_model = MobileNetV2(include_top=False, weights='imagenet',
                                     classes=self.cat_num, input_tensor=input_tensor)
        elif self.model == 'efficientnet':
            base_model = EfficientNetB0(include_top=False, weights='imagenet',
                                        classes=self.cat_num, input_tensor=input_tensor)
        else:
            print('''No model found. Please pass one of the following:
                  inception, resnet, mobilenet, efficientnet''')

        for layer in base_model.layers:
            layer.trainable = False       # freeze layers

        model = Flatten()(base_model.output)
        model = Dense(self.final_layer_neurons, activation=self.activation)(model)
        model = Dense(11, activation='softmax')(model)

        model.compile(loss=self.loss, optimizer='adam',
                      metrics=['accuracy', Precision(), Recall()])
