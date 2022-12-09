import gc
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from keras import Sequential
from keras.metrics import Precision, Recall
from keras.optimizers import Adam

from keras.applications import InceptionV3
from keras.applications import inception_v3
from keras.applications import ResNet50
from keras.applications import resnet
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
                 pooling, X_train, X_test,
                 y_train, y_test, **kwargs):
        self.attribute = attribute
        self.model = model.lower()
        self.input_shape = input_shape
        self.final_layer_neurons = final_layer_neurons
        self.kwargs = kwargs
        self.cat_nums = {
            'design': 7,
            'sleeves': 3,
            'length': 1,    # binary classification
            'neckline': 4,
            'fabric': 6,
            'fit': 3
        }
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=.5
        )
        self.X_test = X_test
        self.y_test = y_test
        del X_train, X_val, X_test, y_train, y_val, y_test
        gc.collect()
        self.cat_num = self.cat_nums[self.attribute]
        self.activation = 'sigmoid' if self.attribute == 'length' else 'softmax'
        self.loss = 'binary_crossentropy' if self.attribute == 'length' else 'categorical_crossentropy'
        self.model = self.instantiate_model()  # calling the function below

    def instantiate_model(self):
        # import ipdb;
        # ipdb.set_trace()
        input = Input(self.input_shape)
        if self.model == 'inception':              # calling the chosen pretrained model
            base_model = InceptionV3(include_top=False, weights='imagenet',
                                    classes=self.cat_num, input_tensor=input)
        elif self.model == 'resnet':
            base_model = ResNet50(include_top=False, weights='imagenet',
                                  classes=self.cat_num, input_tensor=input)
        elif self.model == 'mobilenet':
            base_model = MobileNetV2(include_top=False, weights='imagenet',
                                     classes=self.cat_num, input_tensor=input)
        elif self.model == 'efficientnet':
            base_model = EfficientNetB0(include_top=False, weights='imagenet',
                                        classes=self.cat_num, input_tensor=input)
        else:
            print('''No model found. Please pass one of the following:
                  inception, resnet, mobilenet, efficientnet''')

        base_model.trainable = False    # freeze layers
        pool = GlobalAveragePooling2D() # add pooling layer -> have tried Flatten() here too
        dense = Dense(units=self.final_layer_neurons, activation='relu') # add dense layer
        prediction = Dense(units=self.cat_num, activation=self.activation) # prediction layer
        model = Sequential([base_model, pool, dense, prediction])   # add layers to model
        model.compile(loss=self.loss, optimizer='adam',             # compile model
                      metrics=['accuracy', Precision(), Recall()])
        return model

    def train(self):
        # import ipdb
        # ipdb.set_trace()
        self.model.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.2, verbose=1,
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
        return self.model.history.history

    def finetune(self,
                 metrics=['accuracy', Precision(), Recall()]):
        self.model.trainable = True     # unfreeze layers, then compile to save changes
        self.model.compile(
            optimizer=Adam(1e-5),                  # Very low learning rate
            loss=self.loss,
            metrics=metrics)
        self.model.fit(self.X_val, self.y_val,
                       epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[EarlyStopping(patience=2, monitor='val_accuracy',
                                                restore_best_weights=True)],
                       validation_split=0.2)
        return self.model.history


    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=1)

    def predict(self, X):
        return self.model.predict(X)
