# TODO: implement preprocessing for category and section models


import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator

class Preproc():
    def __init__(self, df: pd.DataFrame, path_img: str, resize_dim: tuple, attr_group: str, section_split: bool) -> None:
        self.df = df
        self.path_img = path_img
        self.resize_dim = resize_dim
        self.attr_group = attr_group
        self._section_split = section_split

        self.attr_names = pd.read_csv('data/list_attr_simple.txt',sep='\s+', header=None).rename(columns={0:'attribute',1:'attribute_type'})
        self.attr_names['attribute_type'] = self.attr_names['attribute_type'].map({1:'design',2:'sleeves',3:'length',4:'part',5:'fabric',6:'fit'})

        self.df_preproc = self._preproc_dataframe()

        self.mean = round(self.df_preproc.iloc[:,1:7].apply(pd.value_counts).iloc[1,:].mean()) # mean number of images in each attribute class

        self.attr_counts = list(self.df_preproc.iloc[:,1:7].apply(pd.value_counts).iloc[1,:].items()) # number of images in each attribute class (list of tuples)

    def _preproc_arrays(self): # create preprocessed arrays
        df_preproc_array = pd.DataFrame() # empty data frame to store augmented and original images as numpy arrays

        for attr, count in self.attr_counts:
            if count >= self.mean: # if number of attribute samples is greater than mean number of samples for the attribute group --> undersample
                sample_df = self.df_preproc[self.df_preproc[attr]==1].sample(self.mean,random_state=2) # sample of images corresponding to mean number of samples for the attribute group
                sample_df['img'] = sample_df.apply(lambda row: self._format_image(row[0], *row[7:]),axis=1) # format (crop and pad) each sampled image and convert to numpy array
                sample_df = sample_df.iloc[:,:7] # only store formatted numpy arrays and attribute columns
            else: # if number of attribute samples is less than mean number of samples for the attribute group --> oversample
                # create list storing number of samples to make for each image:
                sample_values = [0]*count # list storing number of samples for each image
                remaining_sum = self.mean-count # subtract 'count' as original image also converted to numpy array and stored
                i = 0
                while remaining_sum != 0:
                    sample_values[i]+=1
                    remaining_sum-=1
                    i = (i+1)%len(sample_values)

                sample_df = pd.DataFrame() # empty dataframe to store augmented images
                for index, row in enumerate(self.df_preproc[self.df_preproc[attr]==1].values): # iterate over each image beloning to current attribute
                    sample = sample_values[index] # number of samples to be made for current image
                    cropped_img_array = self._format_image(row[0], *row[7:]) # format (crop and pad) image and convert to numpy array
                    temp_oversample_df = pd.concat([pd.DataFrame([row], columns = self.df_preproc.columns).iloc[:,:7]]*(sample+1),ignore_index=True) # dataframe to store oversampled images
                    temp_oversample_df.iloc[0,0] = [cropped_img_array] # store array of original image in dataframe
                    if sample != 0: # if at least one augmentation must be made
                        augmented_img_array = self._augment(cropped_img_array,sample) # augment image 'sample' times and convert each augmentation to numpy array
                        temp_oversample_df.iloc[1:,0] = augmented_img_array # store arrays of augmented images in dataframe
                sample_df = pd.concat([sample_df,temp_oversample_df],axis=0) # store array of original and augmented images
        df_preproc_array = pd.concat([df_preproc_array,sample_df],axis=0)

    def _preproc_dataframe(self): # preprocess the dataframe
        # attribute indexes:
        first_attr = self.attr_names[self.attr_names['attribute_type']==self.attr_group].iloc[0,0] # name of first attribute in selected attributed group
        start_index_attr = np.where(self.df.columns==first_attr) # index of first attribute in selected attribute group within passed dataframe
        indexes_attr = start_index_attr + np.where(self.attr_names['attribute_type'].values=='sleeves')[0] # indexes of attributes within selected attribute group

        # bounding box indexes:
        start_index_bb = np.where(self.df.columns=='x_1') # index of 'x_1' column in passed dataframe

        # dataframe for preprocessing:
        df_preproc = self.df.iloc[:,np.r_[0,indexes_attr[0]:indexes_attr[-1]+1,start_index_bb:start_index_bb+4]] # select all attributes related to 'attr_group' and the image bounding boxes

        return df_preproc

    def _format_image(self, img_name, x1, y1, x2, y2): # crop image by bounding box and resize according to 'resize_dim'
        full_path = self.path_img+img_name # path to image on user's machine

        img = Image.open(full_path) # load images

        cropped = img.crop((x1, y1, x2, y2)) # crop images

        cropped_pad = ImageOps.pad(cropped,self.resize_dim,color=(255,255,255)) # pad image with white background

        cropped_pad_array = np.asarray(cropped_pad)

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
        pass

    def _section_split(self):
        pass
