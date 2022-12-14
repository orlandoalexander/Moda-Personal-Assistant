from fastapi import FastAPI
import numpy as np
import threading
from google.cloud import aiplatform, storage
from keras.applications import efficientnet, mobilenet, resnet
from keras.models import load_model

BUCKET_NAME = "moda-trained-models"
RESIZE_SHAPE = (224, 224)
PROJECT_NUMBER = "530352962003"
MODEL_TYPE = {
    'section': 'mobilenet',
    'landmarks': None,
    'category': 'resnet',
    'sleeves': 'efficientnet',
    'fabric': 'efficientnet',
    'neckline': 'mobilenet',
    'length': 'mobilenet',
    'design': 'efficientnet',
    'fit': 'mobilenet'
}
MODEL_PREPROCESS = {
    'mobilenet': mobilenet.preprocess_input,
    'efficientnet': efficientnet.preprocess_input,
    'resnet': resnet.preprocess_input
}

PRED_CLASSES = {
    "section": ["upper", "lower", "full body", "outfit"],
    "category": [
        "Baggy_Pants", "Blouses", "Cardigans", "Dresses", "Graphic_Tees",
        "Jackets", "Joggers", "Pants", "Rompers", "Shirts", "Shorts", "Skirts",
        "Suiting", "Sweaters", "Tees"
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

app = FastAPI()  # intialise FastAPI object


def get_pad_color(im):
    left = im[:, 0]
    right = im[:, -1]
    edge_color = int(np.concatenate((left, right)).mean())
    mean_color = (edge_color, edge_color)
    return mean_color


def preprocess(im, resize_shape, model_type):
    if im.shape[0] > im.shape[1]:
        scale = (resize_shape[1] - 1) / im.shape[0]
    else:
        scale = (resize_shape[0] - 1) / im.shape[1]
    scale_x, scale_y = (scale * dim for dim in im.shape[:-1])
    x, y = np.ogrid[0:scale_x, 0:scale_y]

    im = im[(x // scale).astype(int), (y // scale).astype(int)]

    if im.shape[0] % 2 == 0:
        ax0_pad_left = ax0_pad_right = int((resize_shape[1] - im.shape[0]) / 2)
    else:
        dif = (resize_shape[1] - im.shape[0])
        ax0_pad_left = int(dif / 2)
        ax0_pad_right = 0
        if dif > 0:
            ax0_pad_right = ax0_pad_left + 1

    if im.shape[1] % 2 == 0:
        ax1_pad_left = ax1_pad_right = int((resize_shape[0] - im.shape[1]) / 2)
    else:
        dif = (resize_shape[0] - im.shape[1])
        ax1_pad_left = int(dif / 2)
        ax1_pad_right = 0
        if dif > 0:
            ax1_pad_right = ax1_pad_left + 1

    pad_color = get_pad_color(im)

    cropped_pad_array = np.pad(
        im,
        pad_width=((ax0_pad_left, ax0_pad_right), (ax1_pad_left,
                                                   ax1_pad_right), (0, 0)),
        constant_values=pad_color)
    cropped_pad_array = cropped_pad_array.reshape((1, cropped_pad_array.shape[0], cropped_pad_array.shape[1], cropped_pad_array.shape[2]))

    return cropped_pad_array



def model_predict(im, model):
    filename = f"{model}/"

    client = storage.Client(project='lewagonbootcamp-371116')
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.download_to_filename(filename)

    model = load_model(filename)


    endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{MODEL_ENDPOINTS[model]}")
    im_formatted = im.astype(np.float32).tolist()



@app.get('/')
def root():
    return {'success': True}


@app.post('/predict')
def predict(im_array):

    im_array_preproc = preprocess(im_array, RESIZE_SHAPE)

    section =

    models = []
    for i in range(0, threads):
        out_list = list()
        thread = threading.Thread(target=list_append(size, i, out_list))
        jobs.append(thread)

    return {'success': True}
