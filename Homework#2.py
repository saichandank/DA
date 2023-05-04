from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from tabulate import tabulate
from numpy.linalg import eig
from numpy.linalg import svd
from prettytable import PrettyTable
from data_preprocessing import *
import math
pd.options.display.float_format = "{:,.2f}".format



#Question 2

yf.pdr_override()
#my_data = yf.download('AAPL', start="2000-01-01", end="2022-09-25")
df = pdr.DataReader('AAPL',start="2000-01-01", end="2022-09-25")

df1 = data_processing(df)
df1.show_originals()
plt.show()
df1.show_normalized()
plt.show()
df1.show_standardized()
plt.show()
df1.show_IQR()
plt.show()

#Question 3
values = np.arange(0.5,6.5,0.5)
colors= ['y','black','red','pink','b','g','orange','purple','olive', 'cyan','grey','m']
a,b = np.meshgrid(np.linspace(-1, 1,num=1000), np.linspace(-1, 1,num=1000))
labels = ['L-0.5 norm','L-1.0 norm','L-1.5 norm','L-2.0 norm','L-2.5 norm','L-3.0 norm','L-3.5 norm','L-4.0 norm','L-4.5 norm','L-5.0 norm','L-5.5 norm','L-6.0 norm']
plt.figure(figsize=(10,10))
plt.grid()
n = 0
for i in values:
  c= ((np.abs((a))**i) + (np.abs((b))**i))**(1./i)
  plt.plot([], [], color=colors[n], label=labels[n])
  plt.contour(a,b,c, levels = [1],colors=colors[n])
  n+=1
plt.legend()
plt.show()

np.random.seed(5525)
#Question 6
x = np.random.normal(1,np.sqrt(2),1000)
e = np.random.normal(2,np.sqrt(3),1000)
df = pd.DataFrame()
df['x']=x
y = x + e
df['y']=y



#Question 6a
X = np.vstack((x, y)).T
mean_X = np.mean(X, axis=0)
cov_X = np.zeros((2, 2))
for i in range(X.shape[0]):
    cov_X += np.outer(X[i] - mean_X, X[i] - mean_X)
cov_X /= X.shape[0]-1
table = PrettyTable()
table.field_names = ["", "x", "y"]
table.add_row(["x", cov_X[0][0], cov_X[0][1]])
table.add_row(["y", cov_X[1][0], cov_X[1][1]])
table.title = "Estimated covariance matrix"
print(table)


#Question 6b
w,v=eig(cov_X)
print('E-value:', w)
print('E-vector', v)
keys = [['Eigen for lambda 1',str(w)],['Eigen for lambda 2',str(v)]]
print(tabulate(keys))

#Question 6c
plt.figure()
plt.scatter(x, y, c= 'r')
plt.plot(v, c= 'b')
plt.title("Scatter plot between x and y")
plt.xlabel('X value')
plt.ylabel('Y value')
plt.grid()
plt.legend(str(v))
plt.show()

#Question 6d
A, B, C = svd(cov_X)
print(A @ B @ C)

#Question 6e
corr = df.corr(method='pearson')
print(corr)

#Question 7

x = np.arange(-4,5,1)
y = [i**3 for i in x]

plt.plot(x,y,'r')
plt.plot(x[:8],np.diff(y,1),'g')
plt.plot(x[:7],np.diff(y,2),'b')
plt.plot(x[:6],np.diff(y,3),'k')
plt.legend(['Original Data','First order','Second order','Third order'])
plt.grid()
plt.show()

first = np.diff(y,1)
second = np.diff(y,2)
third = np.diff(y,3)

q1 = pd.DataFrame()
q1['x(t)'] = x
q1['y(t)'] = y
q2 = pd.DataFrame()
q2['Δy(t)'] = first
q3 = pd.DataFrame()
q3['Δ^2y(t)'] = second
q4 = pd.DataFrame()
q4['Δ^3y(t)'] = third
qM = pd.concat([q1, q2,q3,q4], axis=1)
keys1 = ['x(t)','y(t)','Δy(t)','Δ^2y(t)','Δ^3y(t)']
print(tabulate(qM,headers = keys1, tablefmt = 'fancy_grid'))