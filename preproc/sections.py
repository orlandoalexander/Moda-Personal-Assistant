import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import warnings
from .utils import _format


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

class SectionPreproc():
    def __init__(self, path_img: str, resize_dim: tuple, test_size: float) -> None:
        location = os.path.dirname(os.path.realpath(__file__))
        self._path_img = path_img
        self._resize_dim = resize_dim
        self._test_size = test_size
        self._path_data = os.path.join(location, 'preproc_data')
        self._df = self._get_df()

        self._df_preproc_train, self._df_preproc_test = self._train_test_split()

        self._mean = round(self._df_preproc_train.iloc[:,1:6].apply(pd.value_counts).iloc[1,:].mean()) # mean number of images in each section class in training data set
        self._section_counts = list(self._df_preproc_train.iloc[:,1:6].apply(pd.value_counts).iloc[1,:].items()) # number of images in each section class (list of tuples) in training data set

    def run(self):
        self._X_train, self._y_train = self._preproc_arrays()
        self._df_preproc_test['img'] = self._df_preproc_test.apply(lambda row: self._format_image(row[0]),axis=1)
        self._X_test, self._y_test = self._df_preproc_test.iloc[:,1:6], self._df_preproc_test.iloc[:,0]
        self._y_train, self._y_test = np.array(list(self._y_train)), np.array(list(self._y_test))
        print('Done!')
        return self._X_train, self._y_train, self._X_test, self._y_test

    def _get_df(self): # get formatted data frame
        # Sections:
        section = pd.read_csv(os.path.join(self._path_data,'fabric.txt'),sep=' ',names = ['img','upper_original','lower_original','outer','upper','lower'],header=None,index_col=False).fillna(0)
        section['back'] = np.where(section['img'].str.contains('_back'),1,0)
        section['full body'] = np.where((section['img'].str.contains('WOMEN-Dresses|WOMEN-Rompers')) & ~section['img'].str.contains('_back'),1,0)
        section['outfit'] = np.where((((section['upper_original']!=7) |(section['lower_original']!=7)) & ((section['upper_original']!=7) |(section['outer']!=7))) & (section['full body']==0) & (section['back']==0),1,0)
        section = section.drop(columns=['outer','upper_original','lower_original'])

        # Landmarks:
        landmarks = pd.read_csv(os.path.join(self._path_data,'keypoints_loc.txt'),sep=' ',usecols=[0,15,16,17,18,29,30,31,32,39,40,41,42],names=['img','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','x6','y6'], header=None,index_col=False)

        # Create full data set:
        data_full = section.merge(landmarks, on='img',how='left').dropna()

        # Create upper and lower oberservations (to be cropped later):
        upper = data_full[data_full['outfit']==1]
        lower = data_full[data_full['outfit']==1]
        upper['upper'] = upper['upper'].apply(lambda x: 1)
        upper['outfit'] = upper['outfit'].apply(lambda x: 0)
        lower['lower'] = lower['lower'].apply(lambda x: 1)
        lower['outfit'] = lower['outfit'].apply(lambda x: 0)
        data_full = pd.concat([data_full,upper,lower],axis=0)
        data_full = data_full[(data_full['outfit']==1) | (data_full['full body']==1) | (data_full['back']==1) | (data_full['upper']==1) | (data_full['lower']==1)]

        # Drop invalid rows:
        data_full = data_full[~data_full.isin([-1]).any(1)] # drop rows with inavlid values

        return data_full

    def _preproc_arrays(self): # create preprocessed arrays
        df_preproc_array_df = pd.DataFrame() # empty data frame to store augmented and original images as numpy arrays
        for section, count in self._section_counts:
            if count >= self._mean: # if number of section samples is greater than mean number of samples for the section --> undersample
                print(f"Augmenting section '{section}'...")
                sample_df = self._df_preproc_train[self._df_preproc_train[section]==1].sample(self._mean,random_state=2) # sample of images corresponding to mean number of samples for the section
                if section in ['lower', 'upper']:
                    sample_df['img'] = sample_df.apply(lambda row: self._format_image_outfit(section, row[0], row[6::2].astype(int), row[7::2].astype(int)),axis=1) # split each sampled image into upper/lower, pad and convert to numpy array
                else:
                    sample_df['img'] = sample_df.apply(lambda row: self._format_image(row[0]),axis=1) # pad and convert each image to numpy array
                sample_df = sample_df.iloc[:,:-6] # only store formatted numpy arrays and section columns
            else: # if number of section samples is less than mean number of samples for the section group --> oversample
                print(f"Augmenting section '{section}'...")
                # create list storing number of samples to make for each image:
                sample_values = [0]*int(count) # list storing number of samples for each image
                remaining_sum = self._mean-count # subtract 'count' as original image also converted to numpy array and stored
                i = 0
                while remaining_sum != 0:
                    sample_values[i]+=1
                    remaining_sum-=1
                    i = (i+1)%len(sample_values)
                sample_df = pd.DataFrame() # empty dataframe to store augmented images
                for index, row in enumerate(self._df_preproc_train[self._df_preproc_train[section]==1].values): # iterate over each image beloning to current section
                    sample = sample_values[index] # number of samples to be made for current image
                    if section in ['lower', 'upper']:
                        cropped_img_array = self._format_image_outfit(section, row[0], row[6::2].astype(int), row[7::2].astype(int))
                    else:
                        cropped_img_array = self._format_image(row[0])
                    temp_oversample_df = pd.concat([pd.DataFrame([row], columns = self._df_preproc_train.columns).iloc[:,:-4]]*(sample+1),ignore_index=True) # dataframe to store oversampled images
                    augmented_img_array = []
                    if sample != 0: # if at least one augmentation must be made
                        augmented_img_array = self._augment(cropped_img_array,sample) # augment image 'sample' times and convert each augmentation to numpy array
                    augmented_img_array.append(cropped_img_array)
                    temp_oversample_df.iloc[:,0] = augmented_img_array # store arrays of augmented images in dataframe
                    sample_df = pd.concat([sample_df,temp_oversample_df],axis=0) # store array of original and augmented images
                del augmented_img_array
                del temp_oversample_df

            df_preproc_array_df = pd.concat([df_preproc_array_df,sample_df],axis=0)
            del sample_df

        return df_preproc_array_df.iloc[:,1:6], df_preproc_array_df.iloc[:,0]

    def _format_image(self, img_name):
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = mpimg.imread(full_path) # load images
        img_array = np.asarray(img)
        pad_array = _format(img_array, self._resize_dim).run()

        return pad_array

    def _format_image_outfit(self, section, img_name, x, y):
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = mpimg.imread(full_path) # load images
        if section == 'upper':
            cropped = img[-50+y[0]:y[2], -100+min(x[0],x[1]):max(x[0],x[1])+100]
        if section == 'lower':
            cropped = img[-50+y[2]:y[4], -100+min(x[2],x[3]):max(x[2],x[3])+100]
        cropped_array = np.asarray(cropped)
        cropped_pad_array = _format(cropped_array, self._resize_dim).run() # pad image with white background
        return cropped_pad_array

    def _augment(self, img_array, samples):
        img_array = img_array.reshape((1,) + img_array.shape) # resize image to correct shape

        # Create an ImageDataGenerator object with the desired transformations
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.2,
            rotation_range=15,
            fill_mode='constant', # fill new space created when rotating images with white
            cval=255
        )

        aug_iter = datagen.flow(img_array, batch_size=1) # apply ImageDataGenerator object to sample image array
        arrays = [aug_iter.next()[0].astype('uint8') for i in range (samples)] # create required number of augmented images
        return arrays

    def _train_test_split(self):
        self._df_preproc_train, self._df_preproc_test = train_test_split(self._df, test_size = self._test_size,random_state=2)

        del self._df

        return self._df_preproc_train, self._df_preproc_test
