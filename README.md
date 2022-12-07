# Moda-Personal-Assistant

## Import package

- Works in Google Colab, Jupyter and CLI

1. Make sure `pip` is updated by running `pip install --upgrade pip`
2. Clone the package repo with `git clone -b preproc_package https://github.com/orlandoalexander/Moda-Personal-Assistant.git`
3. Install the package: 
```%cd Moda-Personal-Assistant
! pip install .
```
4. Import the package:
```
from preproc.attributes import Preproc
```
5. Initialise with desired parameters:
```
prep = Preproc(path_to_img_folder, resized_shape_tuple, attribute_group, test_size) # e.g. Preproc(folder/img, (256,256), 'fabric', 0.2)
```
6. Run the preprocessor:
```
X_train, X_test, y_train, y_test = prep.run()
```
