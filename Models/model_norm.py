from tensorflow.keras.applications import inception_v3, resnet, mobilenet_v2, efficientnet

# model arg should be 'inception', 'resnet', 'mobilenet', or 'efficientnet'
class ModelNorm:
    def __init__(self, model, X_train, X_test):
        self.model = model
        self.preprocess_input = {
            'inception': inception_v3.preprocess_input,
            'resnet': resnet.preprocess_input,
            'mobilenet': mobilenet_v2.preprocess_input,
            'efficientnet': efficientnet.preprocess_input
        }
        self.preprocess_x = self.preprocess_input[self.model]
        self.X_train = self.preprocess_x(X_train)
        self.X_test = self.preprocess_x(X_test)
        return X_train, X_test
