#K-Nearest Neighbor Classifier to predict fruits
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
fruits=pd.read_table('fruit.txt')
#checking first five rows of our dataset
print(fruits.head())
# create a mapping from fruit label value to fruit name to make results easier to interpret
predct = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

#checking how many unique fruit names are present in the dataset
Newdf=fruits['fruit_name'].value_counts()
#Now we will store all unique data on four different dataframes.

apple_data=fruits[fruits['fruit_name']=='apple']
orange_data=fruits[fruits['fruit_name']=='orange']
lemon_data=fruits[fruits['fruit_name']=='lemon']
mandarin_data=fruits[fruits['fruit_name']=='mandarin']
plt.plot(mandarin_data['height'],label='Height')
plt.plot(mandarin_data['width'],label='Width')
plt.legend()