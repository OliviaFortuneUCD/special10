#This type of modeling is called regression, hence the "Regressor" part of RandomForestRegressor

#Use a dataset of phone features to predict a phone's price range

#0 (low cost)
#1 (medium cost)
#2 (high cost)
#3 (very high cost)
#The features are things like

#battery_power: Total energy a battery can store in one time measured in mAh
#blue: Has bluetooth or not
#clock_speed: speed at which microprocessor executes instructions
#dual_sim: Has dual sim support or not
#fc: Front Camera mega pixels
#four_g: Has 4G or not

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

data = pd.read_csv('train.csv')
print(data.head())