import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import gc

class SectionPreproc():
    def __init__(self) -> None:
        location = os.path.dirname(os.path.realpath(__file__))
        self._path_data = os.path.join(location, 'preproc_data')

    def _get_df(self): # get formatted data frame
        # Sections:
        section = pd.read_csv(os.path.join(self._path_data,'fabric.txt'),sep=' ',names = ['img','upper','lower','outer'],header=None,index_col=False)
        section['back'] = np.where(section['img'].str.contains('_back'),1,0)
        section['full body'] = np.where(section['img'].str.contains('WOMEN-Dresses|WOMEN-Rompers'),1,0)
        section['outfit'] = np.where((((section['upper']!=7) |(section['lower']!=7)) & ((section['upper']!=7) |(section['outer']!=7))) & (section['full body']==0) & (section['back']==0),1,0)
