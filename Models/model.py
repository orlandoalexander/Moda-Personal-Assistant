from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Input
import pandas as pd
import numpy as np

# For now this class takes in data_full.csv and the path to the images

# In the future it will receive data directly from preprocessing

# Data will be split but here I add a split_data function

DATA_FULL_PATH = '//wsl.localhost/Ubuntu/home/nigel/code/jwnigel/moda-project/code/jwnigel/Moda-Personal-Assistant/Models/data_full.csv'

class Model:
    def __init__(self, attribute,
                data: pd.DataFrame,
                img_shape: tuple):

        # Columns for each attribute are:
        self.attributes = {
                    'design': [np.r_[0,3:10], 7],     # idx 0 are data_full columns
                    'sleeves': [np.r_[0,10:13], 3],   # idx 1 are num attributes
                    'length': [np.r_[0,13:15], 2],
                    'neckline':[np.r_[0,16:20], 4],
                    'fabric': [np.r_[0,20:26], 6],
                    'fit': [np.r_[0,26:29], 3]
                }
        self.data = data[:, attributes[attribute][0]]   # only img and design columns
        self.img_shape = img_shape
        self.num_cats = attributes[attribute][1]
        self.model = self.instatiate_inception()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def split_data(self):
        # Split data into train and test
        self.X_train, self.X_test,
        self.y_train, self.y_test = train_test_split(
            self.data['img'], self.data.drop(columns='img'),
            test_size=0.3, random_state=2)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def instatiate_inception(self):                     # Inception V3 model
        input_layer = Input(shape=(299,299,3))          # Image size (299, 299) specific to Inception V3
        inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_layer)
        inception.layers.trainable = False               # Freeze layers
        model = Sequential(inception)
        model.add(Flatten())
        model.add(Dense(500))                           # Let's play with these last layers
        if self.num_cats == 2:
            model.add(Dense(2, activation='sigmoid'))
        else:
            model.add(Dense(self.num_cats, activation='softmax'))

        model.add(Dense(, activation='softmax'))
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
