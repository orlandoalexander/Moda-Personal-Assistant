# TODO: implement preprocessing for category and section models

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings
from .utils import _format, _augment

from PIL import Image


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class AttributePreproc():
    """
    prep = Preproc(path_to_img_folder_on_laptop, resize_dim_tuple, attribute_group, test_size)
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
        self._attr_names['attribute_type'] = (self._attr_names['attribute_type'].map({1:'design',2:'sleeves',3:'length',4:'neckline',5:'fabric',6:'fit'}))

        self._df_preproc = self._preproc_dataframe()

        self._df_preproc_train, self._df_preproc_test = self._train_test_split()

        del self._df_preproc

        self._mean = round(self._df_preproc_train.iloc[:,1:self._index_range_attr[-1]-self._index_range_attr[0]+2].apply(pd.value_counts).iloc[1,:].mean()) # mean number of images in each attribute class in training data set

        self._attr_counts = list(self._df_preproc_train.iloc[:,1:self._index_range_attr[-1]-self._index_range_attr[0]+2].apply(pd.value_counts).iloc[1,:].items()) # number of images in each attribute class (list of tuples) in training data set
        self._attr_names = [attr[0] for attr in self._attr_counts]

    def run(self):
        self._X_train, self._y_train = self._preproc_arrays()
        self._df_preproc_test['img'] = self._df_preproc_test.apply(lambda row: self._format_image(row[0], *row[-4:]),axis=1)
        self._X_test, self._y_test = self._df_preproc_test.iloc[:,0], np.asarray(self._df_preproc_test.iloc[:,1:-4])
        self._X_train, self._X_test = np.array(list(self._X_train)), np.array(list(self._X_test))
        print('Done!')
        return self._X_train, self._X_test, self._y_train, self._y_test, self._attr_names

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

        # Drop invalid rows:
        if self._attr_group in ['neckline','sleeves']:
            data_full = data_full[data_full['section']!='lower'] # drop all images classified as 'lower'
        if self._attr_group == 'length':
            data_full = (data_full[((data_full['category']=='Dress') & (data_full['no_dress']!=1))]).drop(columns='no_dress') # drop all non-dress images
        return data_full


    def _preproc_dataframe(self): # preprocess the dataframe
        # attribute indexes:
        first_attr = self._attr_names[self._attr_names['attribute_type']==self._attr_group].iloc[0,0] # name of first attribute in selected attributed group
        start_index_attr = np.where(self._df.columns==first_attr)[0] # index of first attribute in selected attribute group within passed dataframe
        self._attr_names.drop(index=12,inplace=True) # drop 'no_dress' attribute
        self._index_range_attr = np.where(self._attr_names['attribute_type'].values==self._attr_group)[0] # range of indexes of attributes within selected attribute group
        end_index_attr = start_index_attr+(self._index_range_attr[-1]-self._index_range_attr[0])+1 # index of final attribute in selected attribute group within passed dataframe

        # bounding box indexes:
        start_index_bb = np.where(self._df.columns=='x_1')[0] # index of 'x_1' column in passed dataframe

        # dataframe for preprocessing:
        df_preproc = self._df.iloc[:,np.r_[0,start_index_attr:end_index_attr,start_index_bb:start_index_bb+4]] # select all attributes related to 'attr_group' and the image bounding boxes

        del self._df

        return df_preproc

    def _preproc_arrays(self): # create preprocessed arrays
        df_preproc_array_df = pd.DataFrame() # empty data frame to store augmented and original images as numpy arrays

        for attr, count in self._attr_counts:
            if count >= self._mean: # if number of attribute samples is greater than mean number of samples for the attribute group --> undersample
                print(f"Augmenting attribute '{attr}'...")
                sample_df = self._df_preproc_train[self._df_preproc_train[attr]==1].sample(self._mean,random_state=2) # sample of images corresponding to mean number of samples for the attribute group
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
                for index, row in enumerate(self._df_preproc_train[self._df_preproc_train[attr]==1].values): # iterate over each image beloning to current attribute
                    sample = sample_values[index] # number of samples to be made for current image
                    cropped_img_array = self._format_image(row[0], *row[-4:]) # format (crop and pad) image and convert to numpy array
                    temp_oversample_df = pd.concat([pd.DataFrame([row], columns = self._df_preproc_train.columns).iloc[:,:-4]]*(sample+1),ignore_index=True) # dataframe to store oversampled images
                    augmented_img_array = []
                    if sample != 0: # if at least one augmentation must be made
                        augmented_img_array = _augment(cropped_img_array,sample, self._pad_color).run() # augment image 'sample' times and convert each augmentation to numpy array
                    augmented_img_array.append(cropped_img_array)
                    temp_oversample_df.iloc[:,0] = augmented_img_array # store arrays of augmented images in dataframe
                    sample_df = pd.concat([sample_df,temp_oversample_df],axis=0) # store array of original and augmented images
                del augmented_img_array
                del temp_oversample_df

            df_preproc_array_df = pd.concat([df_preproc_array_df,sample_df],axis=0)
            del sample_df

        return  df_preproc_array_df.iloc[:,0], np.asarray(df_preproc_array_df.iloc[:,1:])

    def _format_image(self, img_name, x1, y1, x2, y2): # crop image by bounding box and resize according to 'resize_dim'
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = mpimg.imread(full_path) # load images
        cropped = img[y1:y2,x1:x2] # crop images
        cropped_pad_array, self._pad_color = _format(cropped, self._resize_dim).run()

        return cropped_pad_array

    def _train_test_split(self):
        self._df_preproc_train, self._df_preproc_test = train_test_split(self._df_preproc, test_size = self._test_size,random_state=2)
        return self._df_preproc_train, self._df_preproc_test



class SectionPreproc():
    def __init__(self, path_img: str, resize_dim: tuple, test_size: float) -> None:
        location = os.path.dirname(os.path.realpath(__file__))
        self._path_img = path_img
        self._resize_dim = resize_dim
        self._test_size = test_size
        self._path_data = os.path.join(location, 'preproc_data')
        self._df = self._get_df()
        self._df_preproc_train, self._df_preproc_test = self._train_test_split()
        self._mean = round(self._df_preproc_train.iloc[:,1:5].apply(pd.value_counts).iloc[1,:].mean()) # mean number of images in each section class in training data set
        self._section_counts = list(self._df_preproc_train.iloc[:,1:5].apply(pd.value_counts).iloc[1,:].items()) # number of images in each section class (list of tuples) in training data set
        self._section_names = [section[0] for section in self._section_counts]

    def run(self):
        self._X_train, self._y_train = self._preproc_arrays()
        self._df_preproc_test['img'] = self._df_preproc_test.apply(lambda row: self._format_image(row[0]),axis=1)
        self._X_test, self._y_test = self._df_preproc_test.iloc[:,0], np.asarray(self._df_preproc_test.iloc[:,1:6])
        self._X_train, self._X_test = np.array(list(self._X_train)), np.array(list(self._X_test))
        print('Done!')
        return self._X_train, self._X_test, self._y_train, self._y_test, self._section_names

    def _get_df(self): # get formatted data frame
        # Sections:
        section = pd.read_csv(os.path.join(self._path_data,'fabric.txt'),sep=' ',names = ['img','upper_original','lower_original','outer','upper','lower'],header=None,index_col=False).fillna(0)
        section['full body'] = np.where((section['img'].str.contains('WOMEN-Dresses|WOMEN-Rompers')),1,0)
        section['outfit'] = np.where((~section['img'].str.contains('WOMEN-Dresses|WOMEN-Rompers')),1,0)
        #section['outfit'] = np.where((((section['upper_original']!=7) |(section['lower_original']!=7)) & ((section['upper_original']!=7) |(section['outer']!=7))) & (section['full body']==0),1,0)
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
        data_full = data_full[(data_full['outfit']==1) | (data_full['full body']==1) | (data_full['upper']==1) | (data_full['lower']==1)]

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
                    sample_df['img'] = sample_df.apply(lambda row: self._format_image_outfit(section, row[0], row[5::2].astype(int), row[6::2].astype(int)),axis=1) # split each sampled image into upper/lower, pad and convert to numpy array
                else:
                    sample_df['img'] = sample_df.apply(lambda row: self._format_image(row[0]),axis=1) # pad and convert each image to numpy array
                sample_df = sample_df.iloc[:,:-6] # only store formatted numpy arrays and section columns
            else: # if number of section samples is less than mean number of samples for the section --> oversample
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
                        cropped_img_array = self._format_image_outfit(section, row[0], row[5::2].astype(int), row[6::2].astype(int))
                    else:
                        cropped_img_array = self._format_image(row[0])
                    temp_oversample_df = pd.concat([pd.DataFrame([row], columns = self._df_preproc_train.columns).iloc[:,:-4]]*(sample+1),ignore_index=True) # dataframe to store oversampled images
                    augmented_img_array = []
                    if sample != 0: # if at least one augmentation must be made
                        augmented_img_array = _augment(cropped_img_array,sample, self._pad_color).run() # augment image 'sample' times and convert each augmentation to numpy array
                    augmented_img_array.append(cropped_img_array)
                    temp_oversample_df.iloc[:,0] = augmented_img_array # store arrays of augmented images in dataframe
                    sample_df = pd.concat([sample_df,temp_oversample_df],axis=0) # store array of original and augmented images
                del augmented_img_array
                del temp_oversample_df

            df_preproc_array_df = pd.concat([df_preproc_array_df,sample_df],axis=0)
            del sample_df

        return df_preproc_array_df.iloc[:,0], np.asarray(df_preproc_array_df.iloc[:,1:6])

    def _format_image(self, img_name):
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = mpimg.imread(full_path) # load images
        img_array = np.asarray(img)
        pad_array, self._pad_color = _format(img_array, self._resize_dim).run()

        return pad_array

    def _format_image_outfit(self, section, img_name, x, y):
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = mpimg.imread(full_path) # load images

        if section == 'upper':
            cropped = img[-50+min(y[0],y[2]):max(y[0],y[2]), -100+min(x[0],x[1]):max(x[0],x[1])+100]
        if section == 'lower':
            cropped = img[-50+min(y[2],y[4]):max(y[2],y[4]), -100+min(x[2],x[3]):max(x[2],x[3])+100]
        if cropped.size == 0:
            cropped = img
        cropped_array = np.asarray(cropped)
        cropped_pad_array, self._pad_color = _format(cropped_array, self._resize_dim).run() # pad image with appropriate background
        return cropped_pad_array

    def _train_test_split(self):
        self._df_preproc_train, self._df_preproc_test = train_test_split(self._df, test_size = self._test_size,random_state=2)

        del self._df

        return self._df_preproc_train, self._df_preproc_test


class LandmarksPreproc():
    def __init__(self, path_img: str, resize_dim: tuple, test_size: float) -> None:
        location = os.path.dirname(os.path.realpath(__file__))
        self._path_img = path_img
        self._resize_dim = resize_dim
        self._test_size = test_size
        self._path_data = os.path.join(location, 'preproc_data')
        self._df = self._get_df()
        self._df_preproc_train, self._df_preproc_test = self._train_test_split()

    def run(self):
        self._df_preproc_train['img'] = self._df_preproc_train.apply(lambda row: self._format_image(row[0]),axis=1)
        self._df_preproc_test['img'] = self._df_preproc_test.apply(lambda row: self._format_image(row[0]),axis=1)
        self._X_train, self._y_train = self._df_preproc_train.iloc[:,0], np.asarray(self._df_preproc_train.iloc[:,1:])
        self._X_test, self._y_test = self._df_preproc_test.iloc[:,0], np.asarray(self._df_preproc_test.iloc[:,1:])
        self._X_train, self._X_test = np.array(list(self._X_train)), np.array(list(self._X_test))
        print('Done!')
        return self._X_train, self._X_test, self._y_train, self._y_test

    def _get_df(self): # get formatted data frame
        # Sections:
        section = pd.read_csv(os.path.join(self._path_data,'fabric.txt'),sep=' ',names = ['img','upper_original','lower_original','outer','upper','lower'],header=None,index_col=False).fillna(0)
        section['full body'] = np.where((section['img'].str.contains('WOMEN-Dresses|WOMEN-Rompers')) ,1,0)
        section['outfit'] = np.where((((section['upper_original']!=7) |(section['lower_original']!=7)) & ((section['upper_original']!=7) |(section['outer']!=7))) & (section['full body']==0),1,0)
        section = section[section['outfit']==1]
        section = section.drop(columns=['upper','lower','outer','upper_original','lower_original','outfit','full body'])

        # Landmarks:
        landmarks = pd.read_csv(os.path.join(self._path_data,'keypoints_loc.txt'),sep=' ',usecols=[0,15,16,17,18,29,30,31,32,39,40,41,42],names=['img','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','x6','y6'], header=None,index_col=False)

        # Create full data set:
        data_full = section.merge(landmarks, on='img',how='left').dropna()

        # Drop invalid rows:
        data_full = data_full[~data_full.isin([-1]).any(1)] # drop rows with inavlid values
        return data_full

    def _format_image(self, img_name):
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = mpimg.imread(full_path) # load images
        img_array = np.asarray(img)
        pad_array, self._pad_color = _format(img_array, self._resize_dim).run()
        return pad_array

    def _train_test_split(self):
        self._df_preproc_train, self._df_preproc_test = train_test_split(self._df, test_size = self._test_size,random_state=2)

        del self._df

        return self._df_preproc_train, self._df_preproc_test


class CategoryPreproc():
    def __init__(self, path_img: str, resize_dim: tuple, test_size: float) -> None:
        location = os.path.dirname(os.path.realpath(__file__))
        self._path_img = path_img
        self._resize_dim = resize_dim
        self._test_size = test_size
        self._path_data = os.path.join(location, 'preproc_data')
        self._df = self._get_df()
        self._df_preproc_train, self._df_preproc_test = self._train_test_split()

        self._mean = round(self._df_preproc_train.category.value_counts().mean()) # mean number of images in each category class in training data set
        self._section_counts = list(self._df_preproc_train.category.value_counts().items()) # number of images in each category class (list of tuples) in training data set

    def run(self):
        self._X_train, self._y_train, cat_names = self._preproc_arrays()
        self._df_preproc_test['img'] = self._df_preproc_test.apply(lambda row: self._format_image(row[0]),axis=1)
        ohe = OneHotEncoder(sparse=False)
        y_test = np.expand_dims(self._df_preproc_test.iloc[:,-1],axis=1)
        self._y_test = ohe.fit_transform(y_test)

        self._X_test = self._df_preproc_test.iloc[:,0]
        self._X_train, self._X_test = np.array(list(self._X_train)), np.array(list(self._X_test))
        print('Done!')
        return self._X_train, self._X_test, self._y_train, self._y_test, cat_names

    def _get_df(self): # get formatted data frame - only use images of entire model
        # Section:
        section = pd.read_csv(os.path.join(self._path_data,'img.txt'),sep=' ',names = ['img'],header=None,index_col=False).fillna(0)

        # Fabric:
        fabric = pd.read_csv(os.path.join(self._path_data,'fabric.txt'),sep=' ',names = ['img','upper_fabric','lower_fabric'],header=None,index_col=False).fillna(0)
        section = section.merge(fabric,on='img').drop(columns='upper_fabric')

        # Landmarks:
        landmarks = pd.read_csv(os.path.join(self._path_data,'keypoints_loc.txt'),sep=' ',usecols=[0,15,16,17,18,29,30,31,32,39,40,41,42],names=['img','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','x6','y6'], header=None,index_col=False)

        # Category:
        rename_cats = {'Blouses_Shirts':'Blouses',
       'Rompers_Jumpsuits':'Rompers', 'Shirts_Polos':'Shirts', 'Sweatshirts_Hoodies':'Sweaters', 'Tees_Tanks':'Tees',
              'Jackets_Coats':'Jackets', 'Jackets_Vests':'Jackets','Leggings':'Pants','Denim':'Pants'}

        replace_cats = dict()
        with open (os.path.join(self._path_data,'joggers.txt'),'r') as f:
            replace_cats['Joggers'] = f.readlines()[0].split(',')
        with open (os.path.join(self._path_data,'baggy_pants.txt'),'r') as f:
            replace_cats['Baggy_Pants'] = f.readlines()[0].split(',')
        with open (os.path.join(self._path_data,'suiting.txt'),'r') as f:
            replace_cats['Suiting'] = f.readlines()[0].split(',')

        num_files = len(os.listdir(self._path_img))
        img_cats = np.zeros((num_files,2),dtype=object)
        for index,path in enumerate(os.listdir(self._path_img)):
            if path != '.DS_Store' and os.path.isfile(os.path.join(self._path_img, path)):
                img_cat = path.split('-')[1]
                if (img_cat=='Sweaters' or img_cat=='Blouses') and path.split('-')[0]=='MEN':
                    img_cat = 'Shirts'
                replace_cat = [key for key, value in replace_cats.items() if path in value]
                if replace_cat:
                    img_cat = replace_cat[0]
                rename_cat = [value for key, value in rename_cats.items() if img_cat in key]
                if rename_cat:
                    img_cat = rename_cat[0]
                img_cats[index,:]=np.array([path,img_cat])

        img_cats_df = pd.DataFrame(img_cats,columns=['img','category'])
        # Create full data set:
        data_full = section.merge(landmarks, on='img',how='left').merge(img_cats_df, on='img', how='left').dropna()
        data_full.drop(columns='lower_fabric',inplace=True)

        # Drop invalid rows:
        data_full = data_full[~data_full.isin([-1]).any(1)] # drop rows with inavlid values

        return data_full


    def _preproc_arrays(self): # create preprocessed arrays
        df_preproc_array_df = pd.DataFrame() # empty data frame to store augmented and original images as numpy arrays
        for category, count in self._section_counts:
            if count >= self._mean: # if number of category samples is greater than mean number of samples for the category --> undersample
                print(f"Augmenting category '{category}'...")
                sample_df = self._df_preproc_train[self._df_preproc_train['category']==category].sample(self._mean,random_state=2) # sample of images corresponding to mean number of samples for the category
                if category in ['Tees', 'Blouses','Sweaters','Jackets','Cardigans','Graphic_Tees','Shirts']: # if category is of 'upper' type
                    sample_df['img'] = sample_df.apply(lambda row: self._format_image_crop('upper', row[0], row[1:-1:2].astype(int), row[2:-1:2].astype(int)),axis=1) # crop upper half of image, pad and convert to numpy array
                elif category in ['Shorts', 'Pants', 'Skirts', 'Joggers', 'Baggy_Pants']:
                    sample_df['img'] = sample_df.apply(lambda row: self._format_image_crop('lower', row[0], row[1:-1:2].astype(int), row[2:-1:2].astype(int)),axis=1) # crop lower half of image, pad and convert to numpy array
                else:
                    sample_df['img'] = sample_df.apply(lambda row: self._format_image(row[0]),axis=1)
                sample_df = sample_df.iloc[:,np.r_[0,-1]] # only store formatted numpy arrays and category columns
            else: # if number of section samples is less than mean number of samples for the category --> oversample
                print(f"Augmenting category '{category}'...")
                # create list storing number of samples to make for each image:
                sample_values = [0]*int(count) # list storing number of samples for each image
                remaining_sum = self._mean-count # subtract 'count' as original image also converted to numpy array and stored
                i = 0
                while remaining_sum != 0:
                    sample_values[i]+=1
                    remaining_sum-=1
                    i = (i+1)%len(sample_values)
                sample_df = pd.DataFrame() # empty dataframe to store augmented images
                for index, row in enumerate(self._df_preproc_train[self._df_preproc_train.category==category].values): # iterate over each image beloning to current category
                    sample = sample_values[index] # number of samples to be made for current image
                    if category in ['Tees', 'Blouses','Sweaters','Jackets','Cardigans','Graphic_Tees','Shirts']:
                        cropped_img_array = self._format_image_crop('upper', row[0], row[1:-1:2].astype(int), row[2:-1:2].astype(int))
                    elif category in ['Shorts', 'Pants', 'Skirts', 'Joggers', 'Baggy_Pants']:
                        cropped_img_array = self._format_image_crop('lower', row[0], row[1:-1:2].astype(int), row[2:-1:2].astype(int))
                    else:
                        cropped_img_array = self._format_image(row[0])
                    temp_oversample_df = pd.concat([pd.DataFrame([row], columns = self._df_preproc_train.columns).iloc[:,np.r_[0,-1]]]*(sample+1),ignore_index=True) # dataframe to store oversampled images
                    augmented_img_array = []
                    if sample != 0: # if at least one augmentation must be made
                        augmented_img_array = _augment(cropped_img_array,sample, self._pad_color).run() # augment image 'sample' times and convert each augmentation to numpy array
                    augmented_img_array.append(cropped_img_array)
                    temp_oversample_df.iloc[:,0] = augmented_img_array # store arrays of augmented images in dataframe
                    sample_df = pd.concat([sample_df,temp_oversample_df],axis=0) # store array of original and augmented images

                del augmented_img_array
                del temp_oversample_df

            df_preproc_array_df = pd.concat([df_preproc_array_df,sample_df],axis=0)
            del sample_df

        ohe = OneHotEncoder(sparse=False)
        y_train = np.expand_dims(df_preproc_array_df.iloc[:,-1],axis=1)
        y_train = ohe.fit_transform(y_train)
        category_names = [name.replace('x0_','') for name in ohe.get_feature_names_out()]

        return df_preproc_array_df.iloc[:,0], y_train, category_names

    def _format_image(self, img_name):
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = mpimg.imread(full_path) # load images
        img_array = np.asarray(img)
        pad_array, self._pad_color = _format(img_array, self._resize_dim).run()

        return pad_array

    def _format_image_crop(self, section, img_name, x, y):
        full_path = os.path.join(self._path_img,img_name) # path to image on user's machine
        img = mpimg.imread(full_path) # load images
        if section == 'upper':
            cropped = img[-50+y[0]:y[2], -100+min(x[0],x[1]):max(x[0],x[1])+100]
        if section == 'lower':
            cropped = img[-50+y[2]:y[4], -100+min(x[2],x[3]):max(x[2],x[3])+100]
        if cropped.size == 0:
            cropped = img
        cropped_array = np.asarray(cropped)

        cropped_pad_array, self._pad_color = _format(cropped_array, self._resize_dim).run() # pad image with appropriate background
        return cropped_pad_array

    def _train_test_split(self):
        self._df_preproc_train, self._df_preproc_test = train_test_split(self._df, test_size = self._test_size,random_state=2)

        return self._df_preproc_train, self._df_preproc_test
