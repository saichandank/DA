import pandas as pd
from sklearn.datasets import make_blobs,make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

x,y=make_classification(n_samples=1000,n_features=2,n_informative=2,n_repeated=0,n_redundant=0,
                        n_clusters_per_class=2,random_state=5525)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

x1=pd.DataFrame(x_train,columns=['X1','X2'])
y1=pd.DataFrame(y_train,columns=["y1"])
df=pd.concat([x1,y1],axis=1)
sns.scatterplot(data=df,x=df["X1"],y=df["X2"],hue='y1')
plt.grid()
plt.show()

logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

cnf=confusion_matrix(y_test,y_pred)
print(cnf)

from sklearn.metrics import roc_curve,auc
fpr1,tpr1,thresh1=roc_curve(y_test,y_pred)
auc_log=auc(fpr,tpr)
y_prob=logreg.predict_proba(x_test)[::,1]
fpr,tpr,thresh=roc_curve(y_test,y_prob)
auc=auc(fpr,tpr)
plt.plot(fpr,tpr)
plt.show()

b=logreg.intercept_[0]
w1,w2=logreg.coef_.T
c=-b/w1
m=-w1/w2
xmin,xmax=-4,4
xd=np.array([xmin,xmax])
sns.scatterplot(data=df,x=df["X1"],y=df["X2"],hue='y1')





