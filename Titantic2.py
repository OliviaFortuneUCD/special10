import pandas as pd

test_df = pd.read_csv("titantictest.csv")
train_data = pd.read_csv("titantictrain.csv")
train_data.info()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)