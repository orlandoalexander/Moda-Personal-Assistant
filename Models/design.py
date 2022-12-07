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

class Design:

    def __init__(self, data: pd.DataFrame,
              img_path: str):
        self.data = data[:, np.r_[0,3:10]]   # only img and design columns
        self.img_path = img_path           # path to images if passed separately

        # Split data into train and test
        self.X_train, self.X_test,
        self.y_train, self.y_test = train_test_split(
            self.data['img'], self.data.drop(columns='img'),
            test_size=0.3, random_state=2)
        self.model = self.instatiate_inception()

    def instatiate_inception(self):                     # Inception V3 model
        input_layer = Input(shape=(299,299,3))          # Image size (299, 299) specific to Inception V3
        inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_layer)
        inception.layers.trainable = False               # Freeze layers
        model = Sequential(inception)
        model.add(Flatten())
        model.add(Dense(500))                           # Let's play with these last layers
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=[Precision(), Recall()])

    def train(self):
        history = self.model.fit(np.array(self.X_train), self.y_train,
                batch_size=16,
                epochs=50,
                verbose=1,
                callbacks=[EarlyStopping(monitor='val_loss',
                                        patience=5,
                                        restore_best_weights=True)],
                validation_split=0.2)
        return history        # Do I need to return the model?

    def test(self):
        score = self.model.evaluate(np.array(self.X_test), self.y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score

    def predict(self):
        y_pred = self.model.predict(np.array(self.X_test))
        return y_pred
