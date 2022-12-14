import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class _format(): # resize and pad image with appropriate background color
    def __init__(self, image, resize_dim) -> None:
        self._image = image
        self._resize_dim = resize_dim

    def run(self):
        cropped_array = np.asarray(self._image)

        # TEST:
        y_crop = cropped_array.shape[0]
        x_crop = cropped_array.shape[1]
        cropped_array = cropped_array[y_crop//3:y_crop-y_crop//3,x_crop//3:x_crop-x_crop//3,:]

        # 'Zoom' image so either x or y dimensions fits corresponding resize dimensions (or as near as possible)
        if cropped_array.shape[0] > cropped_array.shape[1]:
            scale = (self._resize_dim[1]-1)/cropped_array.shape[0]
        else:
            scale = (self._resize_dim[0]-1)/cropped_array.shape[1]
        scale_x, scale_y = (scale * dim for dim in cropped_array.shape[:-1])
        x, y = np.ogrid[0:scale_x, 0:scale_y]

        cropped_array = cropped_array[(x//scale).astype(int), (y//scale).astype(int)]



        # Pad missing pixels to resize image to require dimensions
        if cropped_array.shape[0] % 2 == 0:
            ax0_pad_left = ax0_pad_right = int((self._resize_dim[1] - cropped_array.shape[0])/2)
        else:
            dif = (self._resize_dim[1] - cropped_array.shape[0])
            ax0_pad_left = int(dif/2)
            ax0_pad_right=0
            if dif > 0:
                ax0_pad_right = ax0_pad_left + 1

        if cropped_array.shape[1] % 2 == 0:
            ax1_pad_left = ax1_pad_right = int((self._resize_dim[0] - cropped_array.shape[1])/2)
        else:
            dif = (self._resize_dim[0] - cropped_array.shape[1])
            ax1_pad_left = int(dif/2)
            ax1_pad_right=0
            if dif > 0:
                ax1_pad_right = ax1_pad_left + 1


        pad_color = self._get_pad_color()

        cropped_pad_array = np.stack([np.pad(cropped_array[:,:,c], ((ax0_pad_left, ax0_pad_right),(ax1_pad_left, ax1_pad_right)), mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)

        del cropped_array

        return cropped_pad_array, pad_color[0]

    def _get_pad_color(self):
        left = self._image[:,0]
        right = self._image[:,-1]
        edge_color = (np.concatenate((left, right)).mean(axis=0))
        return edge_color


class _augment():
    def __init__(self, img_array, samples, pad_color) -> None:
        self._img_array = img_array
        self._samples = samples
        self._pad_color = pad_color

    def run(self):
        self._img_array = self._img_array.reshape((1,) + self._img_array.shape) # resize image to correct shape

        # Create an ImageDataGenerator object with the desired transformations
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.2,
            rotation_range=15,
            fill_mode='constant', # fill new space created when rotating images with white
            cval=self._pad_color
        )

        aug_iter = datagen.flow(self._img_array, batch_size=1) # apply ImageDataGenerator object to sample image array
        arrays = [aug_iter.next()[0].astype('uint8') for i in range (self._samples)] # create required number of augmented images
        return arrays
