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
print(df.head())
df=pre_processing.one_hot(df)
print(df.head())
df= pre_processing.standardize(df)
print(df.head())
df,target=pre_processing.remove_outliers(df,target)
print(df.info())

df_c=df.copy(deep=True)
target_c=target.copy(deep=True)

df,target=pre_processing.data_analysis(df,target)
#df.drop(['table','cut'],axis=1,inplace=True)

pre_processing.correlation(df,"Regression Analysis")
regression_analysis.regression_models(df, target)
df.drop(['y','z','cut'],axis=1,inplace=True)
print(f"Regression model after dropping")
regression_analysis.ols(df, target)
regression_analysis.calc_vif(df)


bins = [0, 3260, 8134, float('inf')]
labels = ['Low', 'Medium', 'High']
target_c = pd.cut(target_c, bins=bins, labels=labels)

pre_processing.feature_selection_classification(df_c,target_c)
df_c.drop(['y','z','table','cut'],axis=1,inplace=True)
#pre_processing.correlation(df_c,"Classification Analysis")
train_model(df_c,target_c)













