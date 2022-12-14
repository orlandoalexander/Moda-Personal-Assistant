from fastapi import FastAPI, UploadFile, File
import numpy as np
import threading
from google.cloud import storage
from keras.applications import efficientnet, mobilenet, resnet
from keras.models import load_model
from PIL import Image

BUCKET_NAME = "moda-trained-models"
RESIZE_SHAPE = (224, 224)
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
# MODEL_ENDPOINTS = {
#     'section': 8725412016728047616,
#     'landmarks': 'TBC',
#     'category': 3258886494030397440,
#     'sleeves': 3420171655685603328,
#     'fabric': 4451214495376736256,
#     'neckline': 3361624860529786880,
#     'length': 1903303006191878144,
#     'design': 3357965685832548352,
#     'fit': 7973310878957174784
# }
CLASS_PREDS = {}
COMPLETE_PREDS = {}

app = FastAPI()  # intialise FastAPI object


def get_pad_color(im):
    left = im[:,0]
    right = im[:,-1]
    edge_color = (np.concatenate((left, right)).mean(axis=0))
    return edge_color


def preprocess(cropped_array, resize_dim, model_type):
    # 'Zoom' image so either x or y dimensions fits corresponding resize dimensions (or as near as possible)
    if cropped_array.shape[0] > cropped_array.shape[1]:
        scale = (resize_dim[1]-1)/cropped_array.shape[0]
    else:
        scale = (resize_dim[0]-1)/cropped_array.shape[1]
    scale_x, scale_y = (scale * dim for dim in cropped_array.shape[:-1])
    x, y = np.ogrid[0:scale_x, 0:scale_y]

    cropped_array = cropped_array[(x//scale).astype(int), (y//scale).astype(int)]

    # Pad missing pixels to resize image to require dimensions
    if cropped_array.shape[0] % 2 == 0:
        ax0_pad_left = ax0_pad_right = int((resize_dim[1] - cropped_array.shape[0])/2)
    else:
        dif = (resize_dim[1] - cropped_array.shape[0])
        ax0_pad_left = int(dif/2)
        ax0_pad_right=0
        if dif > 0:
            ax0_pad_right = ax0_pad_left + 1

    if cropped_array.shape[1] % 2 == 0:
        ax1_pad_left = ax1_pad_right = int((resize_dim[0] - cropped_array.shape[1])/2)
    else:
        dif = (resize_dim[0] - cropped_array.shape[1])
        ax1_pad_left = int(dif/2)
        ax1_pad_right=0
        if dif > 0:
            ax1_pad_right = ax1_pad_left + 1

    pad_color = get_pad_color(cropped_array)
    cropped_pad_array = np.stack([np.pad(cropped_array[:,:,c], ((ax0_pad_left, ax0_pad_right),(ax1_pad_left, ax1_pad_right)), mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    cropped_pad_array = cropped_pad_array.reshape((1,cropped_pad_array.shape[0],cropped_pad_array.shape[1],cropped_pad_array.shape[2]))
    if MODEL_LEARNER[model_type] != None:
        preprocessor = MODEL_PREPROCESS[MODEL_LEARNER[model_type]]
        cropped_pad_array_preproc = preprocessor(cropped_pad_array)
    return cropped_pad_array_preproc


def load_models():
    global LOADED_MODELS
    for model in LOADED_MODELS.keys():
        print(model)
        if model != 'landmarks': # TODO
            LOADED_MODELS[model] = load_model(f"models/{model}/")
    return


def model_predict(im_array, model_type, section):
    global CLASS_PREDS
    global COMPLETE_PREDS
    im_array_preproc = preprocess(im_array, RESIZE_SHAPE, model_type)
    model = LOADED_MODELS[model_type]
    prediction = model.predict(im_array_preproc)

    predicted_classes = [(CLASSES[model_type][index],val) for index, val in enumerate(prediction[0])]
    predicted_classes.sort(key=lambda x: x[1],reverse=True)

    COMPLETE_PREDS[model_type] = predicted_classes

    if section is not None:
        predicted_classes = [pred_class for pred_class in predicted_classes if pred_class[0] in SECTION_CATEGORIES[section]]

    predicted_class = np.argmax(prediction[0], axis=-1)
    predicted_class_name = CLASSES[model_type][predicted_class]

    CLASS_PREDS[model_type] = predicted_class_name

    return predicted_class_name

def predict():
    global CLASS_PREDS
    global COMPLETE_PREDS
    CLASS_PREDS = {}
    COMPLETE_PREDS = {}
    im = Image.open('pred.png')
    im_array = np.asarray(im)

    section = model_predict(im_array, 'section', None)

    if section == 'upper':
        attr_models = ['category','design', 'sleeves', 'neckline', 'fabric', 'fit']

    if section == 'lower':
        attr_models = ['category','design', 'fabric', 'fit']

    if section == 'full body':
        attr_models = ['category','design', 'length', 'sleeves', 'neckline', 'fabric', 'fit']

    if section == 'outfit':
        pass
        # TODO

    for attr_model in attr_models:
        model_predict(im_array, attr_model, section)

    # COLOR TODO

    print(COMPLETE_PREDS)
    return {'results': CLASS_PREDS}


@app.get('/init')
def root():
    load_models()
    return {'message': 'All models successfully loaded'}

@app.get('/update_models')
def update_models(models):
    client = storage.Client(project='lewagonbootcamp-371116')
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()
    for blob in blobs:
        name = blob.name.split('/')
        print(blob.name)
        if (name[1] != '') and name[0] in models:
            file = bucket.blob(blob.name)
            file.download_to_filename(f'models/{blob.name}')
    return {"message": f"Successfully updated all models from Google Cloud"}


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



#return {"message": f"Successfully uploaded {file.filename}"}


# @app.post('/predict')
# def predict():
#     global CLASS_PREDS
#     global COMPLETE_PREDS
#     CLASS_PREDS = {}
#     COMPLETE_PREDS = {}
#     im = Image.open('pred.png')
#     im_array = np.asarray(im)

#     section = model_predict(im_array, 'section', None)

#     if section == 'upper':
#         attr_models = ['category','design', 'sleeves', 'neckline', 'fabric', 'fit']

#     if section == 'lower':
#         attr_models = ['category','design', 'fabric', 'fit']

#     if section == 'full body':
#         attr_models = ['category','design', 'length', 'sleeves', 'neckline', 'fabric', 'fit']

#     if section == 'outfit':
#         pass
#         # TODO

#     for attr_model in attr_models:
#         model_predict(im_array, attr_model, section)

#     # COLOR TODO

#     print(COMPLETE_PREDS)
#     return {'results': CLASS_PREDS}
