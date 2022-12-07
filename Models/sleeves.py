from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Input
import pandas as pd
import numpy as np


PATH = '/content/drive/MyDrive/attribute_img/img/' # IMAGE PATH
FILE_PATH = '/content/drive/MyDrive/txt_files'

class Sleeves:

    def __init__(self, data: pd.DataFrame,
              img_path: str):
                                                                                         # remove lower
        self.data = data[data[self.data['section'] != 'lower']].iloc[:, np.r_[0,10:13]]   # only img and sleeves columns
        self.img_path = img_path           # path to images if passed separately

        # Split data into train and test
        self.X_sleeves_train, self.X_sleeves_test,
        self.y_sleeves_train, self.y_sleeves_test = train_test_split(
            self.data['img'], self.data.drop(columns='img'),
            test_size=0.3, random_state=2)

    def instatiate_inception(self):                     # Inception V3 model
        input_layer = Input(shape=(299,299,3))      # Image size (299, 299) specific to Inception V3
        inception = InceptionV3(include_top=False,       # Don't include fully connected last layer
                                weights='imagenet',      # Weights pretrained on ImageNet
                                input_tensor=input_layer)   # Input shape
        inception.layers.trainable = False                  # Freeze layers so they aren't updated during training

        model = Sequential(inception)
        model.add(Flatten())                      # Let's play with these last layers
        model.add(Dense(500))
        model.add(Dense(3, activation='softmax')) # Would be 4 if including 'lower'

        model.compile(loss='categorical_crossentropy',    # What's the best loss function??
                    optimizer='adam',                   # Try rmseprop too
                    metrics=[Precision(), Recall()])    # Error metrics
        return model

    def train(self, model):
        history = model.fit(np.array(self.X_sleeves_train), self.y_sleeves_train,
                batch_size=16,
                epochs=50,
                verbose=1,
                callbacks=[EarlyStopping(monitor='val_loss',
                                        patience=5,
                                        restore_best_weights=True)],
                validation_split=0.2)
        return model, history        # Do I need to return the model?

    def test(self, model):
        score = model.evaluate(np.array(self.X_sleeves_test), self.y_sleeves_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score

    def predict(self, model):
        y_pred = model.predict(np.array(self.X_sleeves_test))
        return y_pred
