import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pre_processing
import regression_analysis
from  classification_analysis import train_model
from sklearn.metrics import confusion_matrix,auc,roc_curve
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")



# x_train,x_test,y_train,y_test=
df,target=pre_processing.data_analysis()
train_model(df,target)
#print("train data:","\n",x_train.head())


model=SVC()
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)
# confusion_matrix(y_test,y_pred)












