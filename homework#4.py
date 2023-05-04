import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import confusion_matrix,roc_curve,auc

import matplotlib.pyplot as plt
pd.options.display.float_format="{:,0.2f}".format

#question 1
def entropy(x):
    e = 1e-10
    return -x* np.log2(x+e)-(1-x)*np.log2(1-x+e)

def gini(x):
    return 1-(x**2)-(1-x)** 2

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(i) if i!=0 else 0 for i in x ]
gini=[gini(i) for i in x]

plt.figure(figsize=(8, 6))
plt.plot(x, ent, label='Entropy', lw=3)
plt.plot(x, gini, label='Gini Impurity', lw=3)
plt.xlabel('Probability')
plt.ylabel('Impurity')
plt.title('Entropy and Gini Impurity versus Probability')
plt.legend()
plt.show()

#Question 2

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cross_entropy_loss(yHat, y):
    if y == 1:
      return -np.log(yHat)
    else:
      return -np.log(1 - yHat)

z = np.linspace(-10, 10, 100)
sig = sigmoid(z)
cost_1 = cross_entropy_loss(sig, 1)
cost_0 = cross_entropy_loss(sig, 0)


fig, ax = plt.subplots(figsize=(8,6))
plt.plot(sig, cost_1, label='J(w) if y=1', lw=3)
plt.plot(sig, cost_0, label='J(w) if y=0', lw=3, linestyle= '--')
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.title('Log-Loss Function')
plt.tight_layout()
plt.show()

#Question 5

df=sns.load_dataset("titanic")
df.isnull().sum()

df5 = df.select_dtypes(include=['int', 'float'])
df5.fillna(df.mean(numeric_only=True), inplace=True)
X = df5.drop('survived', axis=1)
y = df5['survived']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)


#Decision Tree
tree=DTC(random_state=5525)
tree.fit(x_train,y_train)
y_pred_dt=tree.predict(x_test)
y_prob_dt=tree.predict_proba(x_test)

conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
imps = tree.feature_importances_
idx = np.argsort(imps)[::-1]
print('Importance')
for i in range(x_train.shape[1]):
    print(f'{x_train.columns[idx[i]]}: {imps[idx[i]]}')

#Question 6,7:
tuned_parameters = [{'max_depth': [i for i in range(1, 10)],
                     'min_samples_split': [2, 4, 6, 8, 10],
                     'min_samples_leaf': [i for i in range(1, 10)],
                     'max_features': [4],
                     'splitter': ['best', 'random'],
                     'criterion': ['gini', 'entropy','log_loss']}]

dt_grid = GridSearchCV(tree, param_grid=tuned_parameters, cv=5)
dt_grid.fit(x_train, y_train)

print('Best hyper parameters:', dt_grid.best_params_)

y_pred_dt_gs = dt_grid.predict(x_test)
y_prob_dt_gs=dt_grid.predict_proba(x_test)


conf_matrix_dt_gs=confusion_matrix(y_test,y_pred_dt_gs)


#question 8:
fpr,tpr,_=roc_curve(y_test,y_prob_dt[:,1])
fpr_gs,tpr_gs,_=roc_curve(y_test,y_prob_dt_gs[:,1])

plt.plot(fpr, tpr,color='red', label=f'Before GridSearchAUC: {round(auc(fpr,tpr), 2)}')
plt.plot(fpr_gs, tpr_gs, linestyle='--',color='blue',label=f'After GridSearch AUC: {round(auc(fpr_gs,tpr_gs), 2)}')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend()
plt.show()

sns.heatmap(conf_matrix_dt,annot=True,fmt="0.02f")
plt.title("Confusion matrix before grid search")
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()
plt.show()

sns.heatmap(conf_matrix_dt_gs,annot=True,fmt="0.02f")
plt.title("Confusion matrix after grid search")
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()

#Question 9

log=LR()
log.fit(x_train,y_train)
y_pred_lr=log.predict(x_test)
y_prob_lr=log.predict_proba(x_test)[::,1]
fpr_lr,tpr_lr,_=roc_curve(y_test,y_prob_lr)

cnf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

sns.heatmap(cnf_matrix_lr, annot=True, fmt="0.02f")
plt.title("Confusion matrix Logistic Regression")
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()
plt.show()
auc9 = auc(fpr_lr, tpr_lr)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic regression AUC: {round(auc9, 2)}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

#Question 10:
plt.plot(fpr_gs, tpr_gs, label=f'Grid Search AUC: {round(auc(fpr_gs,tpr_gs),2)}')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic regression AUC: {round(auc(fpr_lr,tpr_lr), 2)}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()








