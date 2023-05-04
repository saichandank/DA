import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tabulate import tabulate
pd.options.display.float_format = "{:,.3f}".format

df=pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Carseats.csv")
#Question 1
#a
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10,6))
df.groupby(['ShelveLoc', 'US'])['Sales'].sum().plot(kind='barh')
plt.xlabel('Sales')
plt.ylabel('Shelve Location')
plt.title('Sales by Shelve Location and US')
plt.legend(title='US', loc='lower right')
plt.show()

#1B
df_one = pd.get_dummies(df, columns=['ShelveLoc', 'Urban', 'US'])
print(df_one.head())

#1C
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_one)
df_scaled = pd.DataFrame(x_scaled, columns=df_one.columns)
train_data,test_data= train_test_split(df_scaled, test_size=0.2, shuffle=True, random_state=1)

print("Training Data:")
print(train_data.head().to_string())
y_train = train_data.pop("Sales")
x_train = train_data

print("Testing Data:")
print(test_data.head().to_string())
y_test = test_data.pop("Sales")
x_test = test_data


#Question 2

def get_stats(x,y):
    model=sm.OLS(y,x).fit()
    return model

selected = list(df_scaled)[1:]
removed=[]
#print(x_col)
y = df_scaled["Sales"]
df_opt=df_scaled.drop("Sales",axis=1)



while(True):
    model = get_stats(df_opt,y)
    max_pvalue = max(model.pvalues)
    if max_pvalue > 0.05:
        index = np.argmax(model.pvalues)
        df_opt =df_opt.drop(selected[index],axis=1)
        #print(model.summary())
        removed.append(selected[index])
        selected.remove(selected[index])
    else:
        break
print(f'\n\nselected features:{selected}\nremoved features:{removed}\n\n')

x_train_1=x_train.drop(removed,axis=1)
x_test_1=x_test.drop(removed,axis=1)
model=get_stats(x_train_1,y_train)
print(model.summary())

y_pred = model.predict(x_test_1)
plt.figure()
plt.title("True Value vs Predicted Value")
plt.plot(y_test, label="True Observation")
plt.plot(y_pred, marker = "*", label = "Predicted Observations")
plt.tight_layout()
plt.legend()
plt.show()

error=np.mean(np.square(np.subtract(y_test,y_pred)))
print(f'\n\nMean Square Error(MSE) =  {error :.3f}\n\n')

#Question3

pca=PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
var_ratio = pca.explained_variance_ratio_
sum =0
count = 0
for i in var_ratio:
  sum+=i
  count+=1
  if sum>0.90:
    print("Number of feature that explains >90% variance: ",count)
    break
l1 = []
l2=[]
sum = 0
for i in range(len(var_ratio)):
  sum = sum+var_ratio[i]
  l1.append(i+1)
  l2.append(sum)

tb = pd.DataFrame(l2,l1)
tb.plot(kind="line",title='Cumulative explained variane Vs Number of features', ylabel='Cumulative explained variance', xlabel='Number of features')
plt.show()

#Question4

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_train)
model = RandomForestClassifier(n_estimators = 100)
model.fit(x_train,y_encoded)
imp = model.feature_importances_
arr = []
sm_arr=[]
for i in range(12):
  sm_arr.append(train_data.columns[i])
  sm_arr.append(f'{imp[i]:.3f}')
  arr.append(sm_arr)
  sm_arr=[]
arr_dec = sorted(arr, key=lambda x: x[1],reverse=True)
name = []
vals = []
for n in arr_dec:
  name.append(n[0])
  vals.append(n[1])
n=name[::-1]
p = vals[::-1]

plt.barh(n, p)
plt.ylabel('Feature Name')
plt.xlabel('Importance')
plt.title('Feature vs Importance')
kl = plt.gca()
kl.invert_yaxis()
plt.show()

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(x_train,y_encoded)

sel_feat= train_data.columns[(sel.get_support())]
removed_feat = train_data.drop((sel_feat),axis=1 )
selected = []
for i in sel_feat:
  selected.append(i)

removed = []

for i in removed_feat:
  removed.append(i)

print("Final Selected Features: ", selected)
print("Eliminated Features: ", removed)

x_train_2 = train_data.drop(removed,axis=1)
x_test_2 = test_data.drop(removed,axis=1)

model = sm.OLS(y_train,x_train_2).fit()
print(model.summary())

y_predicted = model.predict(x_test_2)
error2 = np.mean(np.square(np.subtract(y_test,y_predicted)))


plt.figure()
plt.title("True Value Vs Predicted Value")
plt.plot(y_test, label="True Values")
plt.plot(y_predicted, marker = "*", label = "Predicted Values")
plt.legend()
plt.show()

print(f'\n\nMean Square Error(MSE) =  {error2 :.3f}\n')

#question5

keys = ['Step2', 'Step4']

header = ('R-squared','Adjusted R-Squared', 'AIC', 'BIC', 'MSE' )
header = pd.DataFrame(header)
val1 = (0.874,0.872,253.7,280.1,0.148)
val1 = pd.DataFrame(val1)

val2 = (0.877,0.873,255.4,296.6,0.154)
val2 = pd.DataFrame(val2)
cova = pd.concat([header,val1,val2], axis=1)

print(tabulate(cova,headers=keys, tablefmt = 'fancy_grid', showindex=False))

#Question 6
pred = model.get_prediction(x_test_2)
qs = pred.summary_frame(alpha=0.01)
finaloutput = []
output = []
count = 1
for i in y_predicted:
    output.append(count)
    finaloutput.append(i)
    count += 1
plt.plot(output, finaloutput, label="Predicted Sales", color="b")
plt.xlabel("Number of Samples")
plt.ylabel("Sales USD($)")

lower_ci = qs["mean_ci_lower"]
upper_ci = qs["mean_ci_upper"]

lower = []
upper = []

for i, j in zip(lower_ci, upper_ci):
    lower.append(i)
    upper.append(j)

plt.fill_between(output, lower,upper, color='red', alpha=1, label="CI")
plt.legend()
plt.title("Sales Prediction with Confidence Interval")
plt.show()

#Question 7


training_data, testing_data = train_test_split(df, test_size=0.2, random_state=1,shuffle=True)
y_train = training_data['Sales']
y_test = testing_data['Sales']

X_train_3 = training_data['Price']
X_test_3 = testing_data['Price']

tr_array = np.array(X_train_3)
test_3 = np.array(X_test_3)
print(test_3)
x= []

for i in X_train_3:
  x.append(i)

tr_array=np.array(x)
ar= []

for i in y_train:
  ar.append(i)


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


param_grid = {'polynomialfeatures__degree': np.arange(1,16)}

poly_grid = GridSearchCV(PolynomialRegression(), param_grid, cv=5, scoring= "neg_root_mean_squared_error")
poly_grid.fit(tr_array.reshape(-1,1),y_train)

df1= pd.DataFrame(poly_grid.cv_results_)
print(df1)
print(f'Optimum order of n: {poly_grid.best_params_}')

poly = df1['param_polynomialfeatures__degree']
rmse = df1['mean_test_score']

plt.plot(poly,rmse*-1)
plt.xlabel("Polynomial Order (n)")
plt.ylabel("RMSE")
plt.title("RMSE vs n Order ")
plt.show()

p= PolynomialFeatures(degree = 8)
x_poly = p.fit_transform(tr_array.reshape(-1,1))
x_test_poly = p.fit_transform(test_3.reshape(-1,1))

model = sm.OLS(y_train,x_poly).fit()
y_pr = model.predict(x_test_poly)

op = np.arange(1,81)

plt.plot(op, y_pr, label="Predicted Sales", color="orange")
plt.plot(op, y_test, label = "Test Set", color="blue")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.title("Polynomial Regression model- Sales prediction per the price")
plt.show()

MSE = np.square(np.subtract(y_test,y_pr)).mean()
print(f'Mean Square Error(MSE) =  {MSE :.3f}')