import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression,make_classification,make_blobs
import seaborn as sns

pd.options.display.float_format = "{:,.2f}".format

#Question1
print("Question 1")
samples=1000
features=100
x1,y1=make_regression(n_samples=samples,n_features=features,n_informative=100,n_targets=1,random_state=5525)
print(pd.DataFrame(x1)[:5])
print(f'Target={y1[:5]}')


#Question2
print("Question 2")
x_df=pd.DataFrame(x1,columns=[f'feature{i}'for i in range(1,features+1)])
y_df=pd.DataFrame(y1,columns=["Traget"])

df=pd.concat([x_df,y_df],axis=1)
print(df.tail())

#question 3
print("Question 3")
new_df=df.iloc[0:5,0:5]
print(new_df.to_string())

#question 4
print("Question 4")
cov=np.cov(new_df)
print(cov)
print(new_df.corr())

#Question 5

sns.pairplot(df.iloc[:,:5],kind='kde',diag_kind='kde')
plt.show()

#question 6

plt.scatter(df['feature1'],y_df)
plt.xlabel('feature1')
plt.ylabel('target')
#plt.legend()
plt.show()

#question 7
print("Question 7")
x_c,y_c=make_classification(n_samples=1000,n_features=100,n_informative=100,
                            n_redundant=0,n_classes=4,random_state=5525)
XC=pd.DataFrame(x_c,columns=[f'feature{i}'for i in range(1,101)])
YC=pd.DataFrame(y_c,columns=["Target"])

df_c=pd.concat([XC,YC],axis=1)

print(df_c.head())
print(df_c.tail())

#question 8

slice_df=df_c.iloc[:,:5]
sns.pairplot(slice_df,kind='kde',diag_kind='kde')
plt.show()

#question 9

x_blob,y_blob=make_blobs(n_samples=5000,n_features=2,centers=4,random_state=5525)

df_x=pd.DataFrame(x_blob,columns=[f'feature{i}' for i in range(1,3)])
df_y=pd.DataFrame(y_blob,columns=["Target"])
df_blob=pd.concat([df_x,df_y],axis=1)

#Question 10

sns.scatterplot(data=df_blob,x='feature1',y='feature2',hue='Target')
plt.title("The isotropic gaussian blob with 2 features and 4 centers")
plt.show()



