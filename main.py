import seaborn as sns
import pandas as pd

dataset=sns.get_dataset_names()
print(len(dataset))
df = sns.load_dataset('titanic')
#print(df.describe())  # question 3

print(df.isna().sum())

df1=df.dropna()
print('*'*18)
print("cleaned dataset")

df['deck'].fillna(df['deck'].mode()[0],inplace=True)







