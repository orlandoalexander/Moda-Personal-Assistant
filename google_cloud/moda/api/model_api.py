from fastapi import FastAPI, UploadFile, File
import numpy as np
from keras.applications import efficientnet, mobilenet, resnet
from keras.models import load_model
from PIL import Image
from google.cloud import storage

BUCKET_NAME = "moda-trained-models"
RESIZE_SHAPE = (224, 224)
RESIZE_SHAPE_LANDMARKS = (256,256)
PROJECT_NUMBER = "530352962003"
MODEL_LEARNER = {
    'section': 'mobilenet',
    'landmarks': None,
    'category': 'resnet',
    'sleeves': 'efficientnet',
    'fabric': 'efficientnet',
    'neckline': 'mobilenet',
    'length': 'mobilenet',
    'design': 'resnet',
    'fit': 'mobilenet',
}
LOADED_MODELS = {
    'section': None,
    'landmarks': None,
    'category': None,
    'sleeves': None,
    'fabric': None,
    'neckline': None,
    'length': None,
    'design': None,
    'fit': None,
}
MODEL_PREPROCESS = {
    'mobilenet': mobilenet.preprocess_input,
    'efficientnet': efficientnet.preprocess_input,
    'resnet': resnet.preprocess_input
}
SECTION_CATEGORIES = {
    'upper': ['blouses', 'cardigans', 'graphic_tees', 'jackets', 'shirts', 'suiting', 'sweaters', 'tees'],
    'lower': ['bagg_pants', 'joggers', 'pants', 'shorts', 'skirts', 'suiting'],
    'full body': ['dresses', 'rompers', 'suiting']
}
CLASSES = {
    "section": ["upper", "lower", "full body", "outfit"],
    "category": [
        "baggy_pants", "blouses", "cardigans", "dresses", "graphic_tees",
        "jackets", "joggers", "pants", "rompers", "shirts", "shorts", "skirts",
        "suiting", "sweaters", "tees"
    ],
    "design": [
        "floral", "graphic", "striped", "embroidered", "pleated", "solid",
        "lattice"
    ],
    "sleeves": ["long_sleeve", "short_sleeve", "sleeveless"],
    "length": ["maxi_length", "mini_length"],
    "neckline":
    ["crew_neckline", "v_neckline", "square_neckline"],
    "fabric": ["denim", "chiffon", "cotton", "leather", "faux", "knit"],
    "fit": ["tight", "loose", "conventional"]
}

CLASS_PREDS = {}
COMPLETE_PREDS = {}

app = FastAPI()  # intialise FastAPI object

def get_pad_color(im_array):
    left = im_array[:,0]
    right = im_array[:,-1]
    edge_color = (np.concatenate((left, right)).mean(axis=0))
    return edge_color

def preprocess(im_array, resize_dim):
    # 'Zoom' image so either x or y dimensions fits corresponding resize dimensions (or as near as possible)
    if im_array.shape[0] > im_array.shape[1]:
        scale = (resize_dim[1]-1)/im_array.shape[0]
    else:
        scale = (resize_dim[0]-1)/im_array.shape[1]
    scale_x, scale_y = (scale * dim for dim in im_array.shape[:-1])
    x, y = np.ogrid[0:scale_x, 0:scale_y]

    im_array = im_array[(x//scale).astype(int), (y//scale).astype(int)]

    # Pad missing pixels to resize image to require dimensions
    if im_array.shape[0] % 2 == 0:
        ax0_pad_left = ax0_pad_right = int((resize_dim[1] - im_array.shape[0])/2)
    else:
        dif = (resize_dim[1] - im_array.shape[0])
        ax0_pad_left = int(dif/2)
        ax0_pad_right=0
        if dif > 0:
            ax0_pad_right = ax0_pad_left + 1

    if im_array.shape[1] % 2 == 0:
        ax1_pad_left = ax1_pad_right = int((resize_dim[0] - im_array.shape[1])/2)
    else:
        dif = (resize_dim[0] - im_array.shape[1])
        ax1_pad_left = int(dif/2)
        ax1_pad_right=0
        if dif > 0:
            ax1_pad_right = ax1_pad_left + 1

    pad_color = get_pad_color(im_array)
    cropped_pad_array = np.stack([np.pad(im_array[:,:,c], ((ax0_pad_left, ax0_pad_right),(ax1_pad_left, ax1_pad_right)), mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    cropped_pad_array = cropped_pad_array.reshape((1,cropped_pad_array.shape[0],cropped_pad_array.shape[1],cropped_pad_array.shape[2]))

    return cropped_pad_array


def model_predict(im_array_preproc, model_type, section):
    global LOADED_MODELS
    if MODEL_LEARNER[model_type] != None:
        preprocessor = MODEL_PREPROCESS[MODEL_LEARNER[model_type]]
        im_array_preproc = preprocessor(im_array_preproc)

    model = LOADED_MODELS[model_type]
    prediction = model.predict(im_array_preproc)

    if model_type != 'landmarks':
        predicted_classes = [(CLASSES[model_type][index],val) for index, val in enumerate(prediction[0])]
        predicted_classes.sort(key=lambda x: x[1],reverse=True)

        COMPLETE_PREDS[model_type] = predicted_classes

        if section is not None: # filter attributes to only including attributes which are appropriate for the section
            predicted_classes = [pred_class for pred_class in predicted_classes if pred_class[0] in SECTION_CATEGORIES[section]]

        predicted_class = np.argmax(prediction[0], axis=-1)
        predicted_class_name = CLASSES[model_type][predicted_class]

        return predicted_class_name

    else:
        scaled_coords = prediction[0]/4.30078125 # scale from original size image to scaled images
        x = scaled_coords[::2].astype(int)+40 # account for padding
        y = scaled_coords[1::2].astype(int)

        return x,y


def get_colors():
    pass

def split_outfit(im_array, x, y):
    im_array = im_array.reshape((im_array.shape[1], im_array.shape[2], im_array.shape[3]))
    im_array_lower = im_array[-10+min(y[2],y[4]):max(y[2],y[4]), min(-20+x[2],x[3]):max(x[2],x[3])+20]
    im_array_upper = im_array[-10+min(y[0],y[2]):max(y[0],y[2])+10, -20+min(x[0],x[1]):max(x[0],x[1])+20]
    return im_array_upper, im_array_lower

def predict():
    global CLASS_PREDS
    global COMPLETE_PREDS
    CLASS_PREDS = {}
    COMPLETE_PREDS = {}

    upper_models = ['category','design', 'sleeves', 'neckline', 'fabric', 'fit']
    lower_models = ['category','design', 'fabric', 'fit']

    im = Image.open('pred.png')
    im_array = np.asarray(im)
    im_array_preproc = preprocess(im_array, RESIZE_SHAPE)

    section = model_predict(im_array_preproc, 'section', None)

    if section == 'outfit':
        im_array_preproc_landmarks = preprocess(im_array, RESIZE_SHAPE_LANDMARKS)
        x,y = model_predict(im_array_preproc_landmarks, 'landmarks', None)
        im_array_upper, im_array_lower = split_outfit(im_array_preproc_landmarks, x, y)

        for upper_attr_model in upper_models:
            predicted_class_name=model_predict(im_array_upper, upper_attr_model, 'upper')
            CLASS_PREDS['upper'][upper_attr_model] = predicted_class_name

        for lower_attr_model in lower_models:
            predicted_class_name=model_predict(im_array_lower, lower_attr_model, 'lower')
            CLASS_PREDS['lower'][lower_attr_model] = predicted_class_name

    elif section == 'upper':
        attr_models = upper_models

    elif section == 'lower':
        attr_models = lower_models

    elif section == 'full body':
        attr_models = ['category','design', 'length', 'sleeves', 'neckline', 'fabric', 'fit']

    for attr_model in attr_models:
        predicted_class_name=model_predict(im_array_preproc, attr_model, section)
        CLASS_PREDS[attr_model] = predicted_class_name

    # COLOR TODO

    print(COMPLETE_PREDS)
    return {'results': CLASS_PREDS}

@app.get('/')
def test():
    return {"Success": True}

@app.get('/init')
def update_models():
    global LOADED_MODELS
    storage_client = storage.Client.from_service_account_json('authenticate-gcs.json')
    bucket = storage_client.bucket(BUCKET_NAME)
    for model_name in LOADED_MODELS.keys():
        for file in ['keras_metadata.pb', 'saved_model.pb', 'variables/variables.data-00000-of-00001', 'variables/variables.index']:
            blob = bucket.blob(f"{model_name}/{file}")
            blob.download_to_filename(f'models/{file}')
        model = load_model('models/')
        LOADED_MODELS[model_name] = model
        print(f'{model_name} loaded')
        print(LOADED_MODELS)
    return {'message': 'All models successfully loaded'}

@app.post('/predict')
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open('pred.png', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return predict()
