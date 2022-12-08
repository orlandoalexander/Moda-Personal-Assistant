# Moda-Personal-Assistant

## Import package

- Works in Google Colab, Jupyter and CLI

1. Make sure `pip` is updated:
```
pip install --upgrade pip
```
2. Install the package:
```
pip install git+https://github.com/orlandoalexander/Moda-Personal-Assistant.git@preproc_package
```
3. Import the package from within Python:
```
from preproc.attributes import AttributePreproc
```
4. Initialise with desired parameters:
```
prep = AttributePreproc(path_to_img_folder, resized_shape_tuple, attribute_group, test_size) # e.g. AttributePreproc(folder/img, (256,256), 'fabric', 0.2)
```
5. Run the preprocessor:
```
X_train, X_test, y_train, y_test = prep.run()
```
