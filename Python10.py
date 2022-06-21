#K-Nearest Neighbor Classifier to predict fruits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
fruits=pd.read_table('fruit1.txt')
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
X=fruits[['mass','width','height']]
Y=fruits['fruit_label']
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

print(knn.score(X_test,y_test))
#parameters of following function are mass,width and height
#example1
prediction1=knn.predict([[100,6.3,8]])
print(predct[prediction1[0]])

#example2
prediction2=knn.predict([[300,7,10]])
print(predct[prediction2[0]])

#K-Nearest Neighbors is a supervised learning algorithm. Where the data is 'trained' with data points corresponding to their classification.
# To predict the class of a given data point, it takes into account the classes of the 'K'
#nearest data points and chooses the class in which the majority of the 'K' nearest data points belong to as the predicted class.