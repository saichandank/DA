import numpy as np
from scipy.stats import gmean,hmean
from sklearn import metrics
import matplotlib.pyplot as plt

x=[1,2,3,40,50,10000]
y=np.array(x)
print(f'the arithmatic mean is {np.mean(y):.2f}') # :.2f is for precision
print(f'the harmonic mean is {hmean(y):.2f}')
print(f'the geometric mean is {gmean(y):.2f}')  #comparing three means we can say that there is a outlier if there is a big outlier hm and small gm

#f11 -tp f10-fn f01-fp  f00-tn
#recall vs precision
#f1 score hmean(precision,recall)
np.random.seed(5525)
actual=np.random.binomial(1,0.9,size=1000)
predicted=np.random.binomial(1,0.9,size=1000)

confusion_matrix=metrics.confusion_matrix(actual,predicted)
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=[False,True])
cm_display.plot()
plt.show()
'''TN=6,TP=811,FN=89,FP=94'''

accuracy=metrics.accuracy_score(actual,predicted)
precision=metrics.precision_score(actual,predicted)
recall=metrics.recall_score(actual,predicted)
specificity=metrics.recall_score(actual,predicted,pos_)
f1_score=metrics.f1_score(actual,predicted)
print({'accuracy':accuracy,'precision':precision,'recall':recall,'specificity':specificity,'f1_score':f1_score})


#aggregation groupby()