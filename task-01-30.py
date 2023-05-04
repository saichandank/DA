import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression,make_classification,make_blobs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(5525)
# x1,y1=make_regression(n_samples=1000,n_features=2,n_informative=2,
#                      n_targets=1,bias=0,random_state=5525)
# x1,y1=make_classification(n_samples=1000,n_features=2,n_informative=2,
#                       n_redundant=0,n_classes=2,n_repeated=0,random_state=5525)
# x1=pd.DataFrame(x1,columns=['feature1','feature2'])
# y1=pd.DataFrame(y1,columns=['target'])
#
# df=pd.concat([x1,y1],axis=1)
#
# plt.plot(df['feature1'].values,df['feature2'].values,'o',c='lime',markeredgewidth=0.5,
#          markeredgecolor='black')
# plt.show()
#
# sns.scatterplot(data=df,x='feature1',y='feature2',hue='target')
# plt.show()

x3,y3=make_blobs(n_samples=5000,centers=4,n_features=2,random_state=5525)
x3,y3=pd.DataFrame(x3),pd.DataFrame(y3)
df=pd.concat([x3,y3],axis=1)

sns.scatterplot(data=df,x='feature1',y='feature2',hue='target')
plt.show()
