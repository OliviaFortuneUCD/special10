import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

data = pd.read_csv('train.csv')
# Set variables for the targets and features
y = data['price_range']
X = data.drop('price_range', axis=1)

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=7)
# Create the classifier and fit it to our training data
model = RandomForestClassifier(random_state=7, n_estimators=100)
model.fit(train_X, train_y)
#The simplest metric for classification models is the accuracy,
# the fraction predictions that are correct. Scikit-learn provides metrics.accuracy_score to calculate this.
# Predict classes given the validation features
pred_y = model.predict(val_X)
#Class probabilities
#Classification models actually calculate a probability distribution over the classes. Using model.predict simply returns the class with the highest probability.
#This might not be ideal based on how the decision affects your metrics or downstream measures. To get the probabilities themselves, use the .predict_proba method.

probs = model.predict_proba(val_X)
print(probs)