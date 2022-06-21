#K-Nearest Neighbor Classifier to predict fruits
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
fruits=pd.read_table('fruit.txt')
#checking first five rows of our dataset
print(fruits.head())
# create a mapping from fruit label value to fruit name to make results easier to interpret
predct = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print(predct)
#checking how many unique fruit names are present in the dataset
Newdf=fruits['fruit_name'].value_counts()
print(Newdf)