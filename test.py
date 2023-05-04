import seaborn as sns
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
df1=load_breast_cancer()
df=pd.DataFrame(df1.data,columns=df1.feature_names)
df['target']=pd.Series(df1.tar)

