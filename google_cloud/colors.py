from scipy import cluster
import numpy as np
from fastapi import FastAPI, UploadFile, File
import numpy as np
from keras.applications import efficientnet, mobilenet, resnet
from keras.models import load_model
from PIL import Image
from google.cloud import storage
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
import cv2
import matplotlib.pyplot as plt


def get_pad_color(im_array):
    left = im_array[:,0]
    right = im_array[:,-1]
    edge_color = (np.concatenate((left, right)).mean(axis=0))
    return edge_color

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

    plt.imshow(im_array_preproc)
    plt.show()


    return im_array_preproc

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

def convert_rgb_to_names(rgb_tuple):

    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]

path = '/Users/orlandoalexander/Desktop/images/outfit2.png'
from PIL import Image
im = Image.open(path)
im_array = np.asarray(im)
clusters = 4
sub = 0

x = np.array([111, 156, 120, 153, 134, 128])
y = np.array([ 44,  48, 106, 107, 226, 222])
im_array = preprocess(im_array, (256,256), False)
im_array = preprocess_colors(im_array, x[:-2], y[:-2])


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
color_counts = {code[1]: codebook[code[0]].astype(int) for code in code_counts[:4]}
print(color_counts)
if (np.mean(list(color_counts.values())[0]) -3) < 5:
    sub = list(color_counts.keys())[0]
    color_counts.pop(list(color_counts.keys())[0])
elif (np.mean(list(color_counts.values())[1]) -3) < 5:
    sub = list(color_counts.keys())[1]
    color_counts.pop(list(color_counts.keys())[1])

color_conc = [(convert_rgb_to_names(tuple(color[1])), round((color[0]/(im_array.shape[0]-sub)*100),2)) for color in color_counts.items()]

print(color_conc)
