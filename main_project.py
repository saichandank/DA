import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pre_processing
import regression_analysis
from classification_analysis import train_model

import warnings
warnings.filterwarnings("ignore")



df=pd.read_csv(r"C:\Users\saich\Documents\books\DA\Diamonds Prices2022.csv")
print(df.info())
missing_values=df.isnull().sum()
print(f'missing values in the data set\n{missing_values}')
df.drop(["Unnamed: 0"],axis=1,inplace=True)
target=df['price']
df.drop(['price'],axis=1,inplace=True)
df,target=pre_processing.data_analysis(df,target)

bins = [0, 3260, 8134, float('inf')]
labels = ['Low', 'Medium', 'High']
target = pd.cut(target, bins=bins, labels=labels)
print(target.head(10))


train_model(df,target)













