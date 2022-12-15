from fastapi import FastAPI, UploadFile, File
import numpy as np
from keras.applications import efficientnet, mobilenet, resnet
from keras.models import load_model
from PIL import Image
from google.cloud import storage
from webcolors import hex_to_rgb
from scipy.spatial import KDTree
from scipy import cluster
import cv2

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
COLORS = {'#FF0000':'red','#FFA500':'orange','#FFFF00':'yellow','#90EE90':'light green','#228B22':'forest green', '#00FFFF': 'cyan', '#0000FF':'blue', '#4B0082': 'indigo', '#8F00FF':'violet','#A020F0':'purple','#FFC0CB':'pink','#C0C0C0':'silver','#FFD700':'gold','#F5F5DC':'beige','#6F8FAF':'denim','#800020':'burgundy','#964B00':'brown','#808080':'grey','#000000':'black','#FFFFFF':'white'}

CLASS_PREDS = {}
COMPLETE_PREDS = {}

app = FastAPI()  # intialise FastAPI object

def get_pad_color(im_array):
    left = im_array[:,0]
    right = im_array[:,-1]
    edge_color = (np.concatenate((left, right)).mean(axis=0))
    return edge_color

def preprocess(im_array, resize_dim, reshape):
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
    if reshape:
        cropped_pad_array = cropped_pad_array.reshape((1,cropped_pad_array.shape[0],cropped_pad_array.shape[1],cropped_pad_array.shape[2]))

    return cropped_pad_array


def model_predict(im_array_preproc, model_type, section):
    global LOADED_MODELS
    preprocessor = MODEL_PREPROCESS[MODEL_LEARNER[model_type]]
    im_array_preproc = preprocessor(im_array_preproc)

    model = LOADED_MODELS[model_type]
    prediction = model.predict(im_array_preproc)

    predicted_classes = [(CLASSES[model_type][index],val) for index, val in enumerate(prediction[0])]
    predicted_classes.sort(key=lambda x: x[1],reverse=True)

    COMPLETE_PREDS[model_type] = predicted_classes

    if section is not None: # filter attributes to only including attributes which are appropriate for the section
        predicted_classes = [pred_class for pred_class in predicted_classes if pred_class[0] in SECTION_CATEGORIES[section]]

    predicted_class = np.argmax(prediction[0], axis=-1)
    predicted_class_name = CLASSES[model_type][predicted_class]

    return predicted_class_name


def get_landmarks(im_array_preproc):
    model = LOADED_MODELS['landmarks']
    prediction = model.predict(im_array_preproc)
    scaled_coords = prediction[0]/4.30078125 # scale from original size image to scaled images
    x = scaled_coords[::2].astype(int)+40 # account for padding
    y = scaled_coords[1::2].astype(int)

    return x,y


def preprocess_colors(im_array, x, y):
    pts = (np.array(list(zip(x,y)))).astype('int')
    pts_scaled = pts - pts.min(axis=0) # pts.min(axis=0) gives min pixel in each column

    max_y = np.max(pts_scaled[:,1])
    max_x = np.max(pts_scaled[:,0])
    min_y = np.min(pts_scaled[:,1])
    min_x = np.min(pts_scaled[:,0])


    bl_i = (np.abs(max_y-pts_scaled[:,1]) + np.abs(pts_scaled[:,0]-min_x)).argmin()
    tl_i = (np.abs(min_y-pts_scaled[:,1]) + np.abs(pts_scaled[:,0]-min_x)).argmin()
    tr_i = (np.abs(min_y-pts_scaled[:,1]) + np.abs(pts_scaled[:,0]-max_x)).argmin()
    br_i = (np.abs(max_y-pts_scaled[:,1]) + np.abs(pts_scaled[:,0]-max_x)).argmin()

    bl, tl, tr, br = pts_scaled[bl_i],pts_scaled[tl_i],pts_scaled[tr_i],pts_scaled[br_i]
    pts_ordered = np.array([bl,tl,tr,br])

    pts_ordered = np.array(pts_ordered)

    # Crop the bounding rectangle
    x,y,w,h = cv2.boundingRect(pts)
    cropped = im_array[y:y+h, x:x+w].copy()

    # Make mask
    mask = np.zeros(cropped.shape[:2], np.uint8) # create matrix of zeros with same shape as cropped image
    cv2.drawContours(mask, [pts_ordered], -1, (255, 255, 255), -1, cv2.LINE_AA) # create shape with corners 'pts' on mask image

    # Bitwise and:
    dst = cv2.bitwise_and(cropped, cropped, mask=mask) # bitwise and the cropped image with the mask to keep only pixels within the mask polygon bounds

    # Create white background
    bg = np.ones_like(cropped, np.uint8)*0 # matrix with same shape as cropped image
    cv2.bitwise_not(bg,bg, mask=mask) # white background where mask isn't
    im_array_preproc = bg+dst

    return im_array_preproc



def convert_rgb_to_names(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    names = []
    rgb_values = []
    for color_hex, color_name in COLORS.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]



def get_colors(im_array):
    clusters = 4
    sub = 0
    shape = im_array.shape
    im_array = im_array.reshape(-1, shape[2]).astype(float) # reshape image --> 2D vector
    # Kmeans algorithm on 2D vector to form 'num_clusters':
    codebook, dist_mean = cluster.vq.kmeans(im_array, clusters)
    # 'codebook' stores rgb value for each cluster
    # 'dist_mean' stores mean euclidian distance between each observation and its closest centroid

    # assign code to each observation in image vector 'im_array' (code value corresponds to the rgb color cluster which is closest to the observation)
    codes, dists = cluster.vq.vq(im_array, codebook)
    # 'codes' stores code for each observation in 'im_array' which corresponds to rgb value in 'codebook'
    # 'dists' stores euclidian distance between each observation and its nearest rgb color cluster

    code_counts = sorted({code: np.sum(codes == code) for code in codes}.items(), key=lambda x:x[1],reverse=True)
    color_counts = {code[1]: codebook[code[0]].astype(int) for code in code_counts[:3]}

    if (np.mean(list(color_counts.values())[0]) -3) < 5:
        sub = list(color_counts.keys())[0]
        color_counts.pop(list(color_counts.keys())[0])
    elif (np.mean(list(color_counts.values())[1]) -3) < 5:
        sub = list(color_counts.keys())[1]
        color_counts.pop(list(color_counts.keys())[1])
    print()
    color_conc = [(convert_rgb_to_names(tuple(color[1])), round((color[0]/(im_array.shape[0]-sub)),2)) for color in color_counts.items()]
    print(color_conc)
    color1 = color_conc[0][0]
    if color_conc[0][1]>=0.8:
        color2 = color3 = color1
    else:
        color2 = color_conc[1][0]
    if color_conc[2][1] > 0.1:
        color3 = color_conc[1][0]
    else:
        color3 = color2

    return (color1,color2, color3)


def split_outfit(im_array, x, y):
    im_array = im_array.reshape((im_array.shape[1], im_array.shape[2], im_array.shape[3]))
    try:
        im_array_lower = im_array[-10+min(y[2],y[4]):max(y[2],y[4])+10, -30+min(x[2],x[3]):max(x[2],x[3])+30]
    except:
        im_array_lower = im_array[min(y[2],y[4]):max(y[2],y[4]), min([2],x[3]):max(x[2],x[3])]
    im_array_lower = preprocess(im_array_lower, RESIZE_SHAPE, True)
    try:
        im_array_upper = im_array[-10+min(y[0],y[2]):max(y[0],y[2])+10, -30+min(x[0],x[1]):max(x[0],x[1])+30]
    except:
        im_array_upper = im_array[min(y[0],y[2]):max(y[0],y[2]), min(x[0],x[1]):max(x[0],x[1])]

    im_array_upper = preprocess(im_array_upper, RESIZE_SHAPE, True)
    return im_array_upper, im_array_lower



def predict():
    global CLASS_PREDS
    global COMPLETE_PREDS
    CLASS_PREDS = {}
    COMPLETE_PREDS = {}

    upper_models = ['category','design', 'sleeves', 'neckline', 'fabric', 'fit']
    lower_models = ['category','design', 'fabric', 'fit']
    full_models = ['category','design', 'length', 'sleeves', 'neckline', 'fabric', 'fit']

    im = Image.open('pred.png')
    im_array = np.asarray(im)
    im_array_preproc = preprocess(im_array, RESIZE_SHAPE, True)
    im_array_preproc_colors = preprocess(im_array, RESIZE_SHAPE, False)
    im_array_preproc_landmarks = preprocess(im_array, RESIZE_SHAPE_LANDMARKS, True)

    x,y = get_landmarks(im_array_preproc_landmarks)
    #x = np.array([111, 156, 120, 153, 134, 128])
    #y = np.array([ 44,  48, 106, 107, 226, 222])
    print(x,y)

    section = model_predict(im_array_preproc, 'section', None)
    if section == 'outfit':
        im_array_upper, im_array_lower = split_outfit(im_array_preproc_landmarks, x, y)

        # TODO
        im = Image.fromarray(im_array_upper.reshape((224,224,3)))
        im.save('upper.png')
        im = Image.fromarray(im_array_lower.reshape((224,224,3)))
        im.save('lower.png')


        CLASS_PREDS['upper'] = {}
        CLASS_PREDS['lower'] = {}
        for upper_attr_model in upper_models:
            predicted_class_name=model_predict(im_array_upper, upper_attr_model, 'upper')
            CLASS_PREDS['upper'][upper_attr_model] = predicted_class_name
        im_array_preproc_colors = preprocess_colors(im_array_preproc_colors, x[:-2], y[:-2])
        colors_upper = get_colors(im_array_preproc_colors)
        CLASS_PREDS['upper']['colors'] = colors_upper

        for lower_attr_model in lower_models:
            predicted_class_name=model_predict(im_array_lower, lower_attr_model, 'lower')
            CLASS_PREDS['lower'][lower_attr_model] = predicted_class_name
        im_array_preproc_colors = preprocess_colors(im_array_preproc_colors, x[2:], y[2:])
        colors_lower = get_colors(im_array_preproc)
        CLASS_PREDS['lower']['colors'] = colors_lower
    else:
        if section == 'upper':
            attr_models = upper_models

        if section == 'lower':
            attr_models = lower_models

        if section == 'full body':
            attr_models = full_models

        for attr_model in attr_models:
            predicted_class_name=model_predict(im_array_preproc, attr_model, section)
            CLASS_PREDS[attr_model] = predicted_class_name
        colors = get_colors(im_array_preproc)
        CLASS_PREDS['colors'] = colors

    print(COMPLETE_PREDS)
    return {'results': CLASS_PREDS}

@app.get('/')
def test():
    return {"Success": True}

@app.get('/init')
def update_models():
    global LOADED_MODELS
    # storage_client = storage.Client.from_service_account_json('authenticate-gcs.json')
    # bucket = storage_client.bucket(BUCKET_NAME)
    # for model_name in LOADED_MODELS.keys():
    #     for file in ['keras_metadata.pb', 'saved_model.pb', 'variables/variables.data-00000-of-00001', 'variables/variables.index']:
    #         blob = bucket.blob(f"{model_name}/{file}")
    #         blob.download_to_filename(f'models/{file}')
    #     model = load_model('models/')
    #     LOADED_MODELS[model_name] = model
    import os
    for i in os.listdir('models'):
        if i != '.DS_Store':
            model = load_model('models/'+i)
            LOADED_MODELS[i.split('_')[1]] = model
            print(f"{i.split('_')[1]} loaded")
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
