import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
pd.options.display.float_format = "{:,.2f}".format

#Question 1
url="https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/stock%20prices.csv"
df = pd.read_csv(url)
print(df.isnull().sum())
df['open'].fillna(df['open'].mean(),inplace=True)
df['high'].fillna(df['high'].mean(),inplace=True)
df['low'].fillna(df['low'].mean(),inplace=True)
print(df.isnull().sum())

#Question2
print("number of unique companies:",len(df['symbol'].unique()),"\n List of companies",df['symbol'].unique())
quantitative_predictors = []
qualitative_predictors = []

for column in df.columns:
    if df[column].dtype == 'int64' or df[column].dtype == 'float64':
        quantitative_predictors.append(column)
    else:
        qualitative_predictors.append(column)
print('Quantitative predictors: ', quantitative_predictors)
print('Qualitative predictors: ', qualitative_predictors)

df1 = df.loc[df['symbol'].isin(['AAPL', 'GOOGL'])]
df1g=df.loc[df['symbol'].isin(['GOOGL'])]
df1a=df.loc[df['symbol'].isin(['AAPL'])]

plt.plot(pd.to_datetime(df1a['date']),df1a['close'],label='apple')
plt.plot(pd.to_datetime(df1g['date']),df1g['close'],label='google')
plt.xlabel("DATE")
plt.ylabel("STOCK VALUE")
plt.title("Stock value comparision between google and apple")
plt.legend()
plt.grid(axis='y')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=8))
plt.figure(figsize=(12,8))
plt.tight_layout()
plt.show()

#question 3
df3 = df.groupby('symbol').sum()
print(df3.head())
print(f'\n The number of objects in the aggragated df is : {len(df3)}')
print(f'\n The number of objects in the orginal df is :{len(df)}')

#Question 4
df1 = df.loc[:497472,['symbol','close', 'volume']]
df4 = df1.groupby(['symbol']).agg([np.mean, np.var])
max1 = df4['close'].idxmax()
print(df4.head())
print(f'The company with maximum variance for closing is :\n{max1}')

#Question 5
df5 = df.loc[df['symbol'].isin(['GOOGL'])]
df5.drop(['open', 'high', 'low','volume'],axis=1)
df5 = df5[df5['date'] >= '2015-01-01']
print(df5.head(5))

# df6 = pd.DataFrame(df5.close)
df5['AVG_30'] = df5.close.rolling(window=30, center=True).aggregate(np.mean)

plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(df5)+1),df5['AVG_30'].values, label = 'AVG_30-Filtered', lw = 4.5)
plt.plot(df5['close'].values, label = 'Close-Original', lw = 2.5)
plt.legend()
plt.xlabel('Date')
plt.ylabel('USD($)')
plt.title('Google closing stock price after Jan 2015 vs Rolling window')
plt.grid(axis='y')
plt.show()

#Question 7

df6 = pd.DataFrame(data = df5,columns=['close'])
df6['group'] = pd.cut(df6['close'], bins = 5, labels = ['very low', 'low','normal', 'high', 'very high'])
df6['group'] = df6['group'].astype('object')
plt.figure()
sns.countplot(x = 'group', data = df6)
plt.title("Equal Width Discretization")
plt.xlabel('Price Category')
plt.grid(axis = 'y')
plt.tight_layout()
plt.show()
print(df6.to_string())

#Question 8
plt.hist(df6['close'],bins=5,label='close')
plt.title('Histogram of Close feature')
plt.show()

#Question 9
df7 = pd.DataFrame(data = df5,columns=['close'])
df7['group'] = pd.qcut(df7['close'], q= 5,precision=0, labels = ['very low', 'low','normal', 'high', 'very high'])
df7.group = df7.group.astype('object')
plt.figure()
sns.countplot(x = 'group', data = df7)
plt.xlabel('Price Category')
plt.title("Equal Frequency Disrcetization")
plt.grid(axis = 'y')
plt.tight_layout()
plt.show()
print(df7.to_string())

#Question 10
df_filtered = df[(df['symbol'] == 'GOOGL') & (df['date'] >= '2015-01-01')]
mean = df_filtered[['open', 'high', 'low', 'close', 'volume']].mean()
centered_data = df_filtered[['open', 'high', 'low', 'close', 'volume']] - mean
covariance_matrix = np.dot(centered_data.T, centered_data) / (len(df_filtered) - 1)
print("Covariance matrix:\n", covariance_matrix)

#Question 11
cov_matrix = df_filtered.drop("date", axis=1).cov()
print("Covariance Matrix:")
print(cov_matrix)

