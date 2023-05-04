import statsmodels.api as sm
from sklearn.feature_selection import f_classif
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def t_test(df,target):
    print(f'*******************T-test******************')
    for i in df.columns:
        tvalue,_=stats.ttest_rel(df[i],target)
        print(f'\nT-value for {i} vs target: {tvalue,_}')

def ftest(df,target):
    f_value,p_value=f_classif(df,target)
    print("***********F-Test***********************")
    print(f'f_value={f_value}\np_value={np.round(p_value,2)}')
    print("****************************************")

def ols(df,target):
    df=sm.add_constant(df)
    results = sm.OLS(target,df).fit()
    print(f' Original OLS: \n {results.summary()}')


def calc_vif(df):

    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    print(vif)

def regression_models(df,target):
    t_test(df,target)
    ols(df,target)
    ftest(df,target)
    calc_vif(df)
