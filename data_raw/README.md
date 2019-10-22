# data_raw 
- raw data without pre-processing, as is, is stored here. 

## File Description (from Kaggle)
- train_max_x - Contains the training set images. 
- train_max_y.csv - Contains labels for the training set. The data contains two fields: Id and Label
- test_max_x - Contains the test set images.

## Data Field Descriptions (from Kaggle)
- Id - An unique integer associated with every image
- Label - maximum of the 3 digits from the image.

## How to read training and test test image (from Kaggle)
import pandas as pd train_images = pd.read_pickle('train_max_x') test_images = pd.read_pickle('test_max_x')
