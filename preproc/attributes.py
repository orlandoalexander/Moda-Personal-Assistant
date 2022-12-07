# TODO: implement preprocessing for category and section models
# TODO: add feature preprocessing depending on the attribute type

import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import gc


class Preproc():
    """
    prep = Preproc(data_full, path_to_img_folder_on_laptop, resize_dim_tuple, attribute_group, test_size)
    X_train, X_test, y_train, y_test = prep.run()
    """
    def __init__(self, path_img: str, resize_dim: tuple, attr_group: str, test_size: float):
        location = os.path.dirname(os.path.realpath(__file__))
        self._path_img = path_img
        self._resize_dim = resize_dim
        self._attr_group = attr_group.lower()
        self._test_size = test_size
        self._path_data = os.path.join(location, 'preproc_data')
        self._df = self._get_df()

        self._attr_names = pd.read_csv(os.path.join(self._path_data,'list_attr_simple.txt'),sep='\s+', header=None).rename(columns={0:'attribute',1:'attribute_type'})
        self._attr_names['attribute_type'] = self._attr_names['attribute_type'].map({1:'design',2:'sleeves',3:'length',4:'neckline',5:'fabric',6:'fit'})

        self._df_preproc = self._preproc_dataframe()

        self._mean = round(self._df_preproc.iloc[:,1:-4].apply(pd.value_counts).iloc[1,:].mean()) # mean number of images in each attribute class

        self._attr_counts = list(self._df_preproc.iloc[:,1:-4].apply(pd.value_counts).iloc[1,:].items()) # number of images in each attribute class (list of tuples)

    def run(self):
        self._X_train, self._y_train = self._preproc_arrays()
        self._y_train = np.array(list(self._y_train))
        return self._train_test_split()

    def _get_df(self): # get formatted data frame
        # Categories:
        cat_names = pd.read_csv(os.path.join(self._path_data,'list_category.txt'),sep ='\s+',header=None).reset_index().rename(columns={0:'category',1:'section','index':'cat_num'})
        cat_names['cat_num'] = cat_names['cat_num'].apply(lambda x : x+1)
        cat_tag = pd.read_csv(os.path.join(self._path_data,'tag_cat_simple.txt'),sep='\s+',header=None,names=['category']).rename(columns={'category':'cat_num'})
        cat_img = pd.read_csv(os.path.join(self._path_data,'img_simple.txt'),sep='\s+',header=None,names=['category']).rename(columns={'category':'img'})
        cat_num = cat_img.join(cat_tag)
        cat = cat_num.merge(cat_names, on='cat_num', how='left').drop(columns='cat_num')
        cat['img'] = cat['img'].apply(lambda x: x[4:])
        cat['section'] = cat['section'].map({1:'upper',2:'lower',3:'full body'})

        # Attributes:
        attr_names = pd.read_csv(os.path.join(self._path_data,'list_attr_simple.txt'),sep='\s+', header=None).rename(columns={0:'attribute',1:'attribute_type'})
        attr_names['attribute_type'] = attr_names['attribute_type'].map({1:'design',2:'sleeves',3:'length',4:'neckline',5:'fabric',6:'fit'})
        attr_names_headers = attr_names.iloc[:,0]
        attr_tags = pd.read_csv(os.path.join(self._path_data,'tags_attr_simple.txt'),sep='\s+',header=None,names=attr_names_headers)
        attr_img = pd.read_csv(os.path.join(self._path_data,'img_simple.txt'),sep='\s+',header=None).rename(columns={0:'img'})
        attr = attr_img.join(attr_tags,how='inner').drop(columns='img')

        # Bounding Boxes:
        bb = pd.read_csv(os.path.join(self._path_data,'bbox.txt'),sep='\s+',header=None,index_col=False, names=['img','x_1', 'y_1', 'x_2', 'y_2'])
        bb['img'] = bb['img'].apply(lambda x: x[4:])

        # Landmarks:
        landmarks = pd.read_csv(os.path.join(self._path_data,'landmarks.txt'),sep='\s+',names=['img','clothes_type','v1','x1','y1','v2','x2','y2','v3','x3','y3','v4','x4','y4','v5','x5','y5','v6','x6','y6','v7','x7','y7','v8','x8','y8']).fillna(0).drop(columns=['clothes_type'])
        landmarks['img'] = landmarks['img'].apply(lambda x: x[4:])

        # Create full data set:
        data_full = cat.join(attr,how='left').merge(bb,how='left',on='img').merge(landmarks, how='left',on='img')

        return data_full


    def _preproc_dataframe(self): # preprocess the dataframe
        # attribute indexes:
        first_attr = self._attr_names[self._attr_names['attribute_type']==self._attr_group].iloc[0,0] # name of first attribute in selected attributed group
        start_index_attr = np.where(self._df.columns==first_attr)[0] # index of first attribute in selected attribute group within passed dataframe
        index_range_attr = np.where(self._attr_names['attribute_type'].values==self._attr_group)[0] # range of indexes of attributes within selected attribute group
        end_index_attr = start_index_attr+(index_range_attr[-1]-index_range_attr[0])+1 # index of final attribute in selected attribute group within passed dataframe

        # bounding box indexes:
        start_index_bb = np.where(self._df.columns=='x_1')[0] # index of 'x_1' column in passed dataframe

        # dataframe for preprocessing:
        df_preproc = self._df.iloc[:,np.r_[0,start_index_attr:end_index_attr,start_index_bb:start_index_bb+4]] # select all attributes related to 'attr_group' and the image bounding boxes

        return df_preproc

    def _preproc_arrays(self): # create preprocessed arrays
        df_preproc_array_df = pd.DataFrame() # empty data frame to store augmented and original images as numpy arrays

        for attr, count in self._attr_counts:
            if count >= self._mean: # if number of attribute samples is greater than mean number of samples for the attribute group --> undersample
                print(f"Augmenting attribute '{attr}'...")
                sample_df = self._df_preproc[self._df_preproc[attr]==1].sample(self._mean,random_state=2) # sample of images corresponding to mean number of samples for the attribute group
                sample_df['img'] = sample_df.apply(lambda row: self._format_image(row[0], *row[-4:]),axis=1) # format (crop and pad) each sampled image and convert to numpy array
                sample_df = sample_df.iloc[:,:-4] # only store formatted numpy arrays and attribute columns
            else: # if number of attribute samples is less than mean number of samples for the attribute group --> oversample
                print(f"Augmenting attribute '{attr}'...")
                # create list storing number of samples to make for each image:
                sample_values = [0]*int(count) # list storing number of samples for each image
                remaining_sum = self._mean-count # subtract 'count' as original image also converted to numpy array and stored
                i = 0
                while remaining_sum != 0:
                    sample_values[i]+=1
                    remaining_sum-=1
                    i = (i+1)%len(sample_values)
                sample_df = pd.DataFrame() # empty dataframe to store augmented images
                for index, row in enumerate(self._df_preproc[self._df_preproc[attr]==1].values): # iterate over each image beloning to current attribute
                    sample = sample_values[index] # number of samples to be made for current image
                    cropped_img_array = self._format_image(row[0], *row[-4:]) # format (crop and pad) image and convert to numpy array
                    temp_oversample_df = pd.concat([pd.DataFrame([row], columns = self._df_preproc.columns).iloc[:,:-4]]*(sample+1),ignore_index=True) # dataframe to store oversampled images
                    augmented_img_array = []
                    if sample != 0: # if at least one augmentation must be made
                        augmented_img_array = self._augment(cropped_img_array,sample) # augment image 'sample' times and convert each augmentation to numpy array
                    augmented_img_array.append(cropped_img_array)
                    temp_oversample_df.iloc[:,0] = augmented_img_array # store arrays of augmented images in dataframe
                    sample_df = pd.concat([sample_df,temp_oversample_df],axis=0) # store array of original and augmented images
                del augmented_img_array
                del temp_oversample_df
                gc.collect()
            df_preproc_array_df = pd.concat([df_preproc_array_df,sample_df],axis=0)
            del sample_df
            gc.collect()
        return df_preproc_array_df.iloc[:,1:], df_preproc_array_df.iloc[:,0]


    def _format_image(self, img_name, x1, y1, x2, y2): # crop image by bounding box and resize according to 'resize_dim'
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = Image.open(full_path) # load images
        cropped = img.crop((x1, y1, x2, y2)) # crop images
        cropped_pad = ImageOps.pad(cropped,self._resize_dim,color=(255,255,255)) # pad image with white background
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
        self._X_train, X_test, self._y_train, y_test = train_test_split(self._X_train, self._y_train, test_size = self._test_size,random_state=2)
        print('Done!')
        return self._X_train, X_test, self._y_train, y_test
