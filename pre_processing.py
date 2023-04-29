import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import regression_analysis

def one_hot(df):
    #print(df['cut'].unique())
    df['cut']=df['cut'].replace(['Ideal','Premium','Very Good','Good','Fair'],[1,2,3,4,5])
    df1=df[['color','clarity']]
    df1=pd.get_dummies(df1)
    df.drop(['color','clarity'],axis=1,inplace=True)
    target=df['cut']
    df.drop(['cut'],axis=1,inplace=True)
    df.head()
    return df,df1,target

def standardize(df,df1):
    scalar = StandardScaler()
    values = scalar.fit_transform(df)
    df_scalar = pd.DataFrame(values, columns=df.columns)
    df_scalar.boxplot()
    plt.show()
    #print(df_scalar.head())
    df = pd.concat([df, df1], axis=1)
    #print(df.shape, "\n", df.head())
    return df

def remove_outliers(df):
    pass
def feature_selection(df,target):
    x_train,x_test,y_train,y_test=train_test_split(df,target,test_size=0.2,shuffle=True,random_state=5525)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_
    feature_names = df.columns
    sorted_idx = importances.argsort()
    print("Feature importances:")
    for i in range(0,len(importances)):
        print(feature_names[i], importances[i])

    #*********PCA Analysis************

    print('\n **********************************')
    x_scaled=StandardScaler().fit_transform(df)
    pca = PCA()
    X_pca = pca.fit_transform(x_scaled)
    e_values=np.round(pca.explained_variance_,2)
    variance_ratio=pca.explained_variance_ratio_
    print(f'Variance explained by all  principal components = {sum(variance_ratio* 100):.2f}')
    print(np.cumsum(variance_ratio * 100))

    plt.figure()
    plt.plot(np.cumsum(variance_ratio*100))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.title('Elbow plot for PCA')
    plt.show()

    print("Variance explained by the First principal component=", np.cumsum(variance_ratio[0]))
    print("Variance explained by the First 2 principal component=",np.cumsum(variance_ratio[1]))
    print("Variance explained by the First 3 principal component=",np.cumsum(variance_ratio[2]))
    print("Variance explained by the First 7 principal component=",np.cumsum(variance_ratio[6]))

    print('\n **********************************')

    #***************SVD********************************

    svd = TruncatedSVD(n_components=7)
    X_svd = svd.fit_transform(df)
    print(f'svd:{X_svd.shape}\n{pd.DataFrame(X_svd).head()}')

#************************Covariance Matrix*****************************
def correlation(df):
    print('\n **********************************')
    print(f' Covariance Matrix is: \n {df.cov().to_string()}')
    cov_heat = df.cov()
    sns.heatmap(cov_heat, cmap="crest", vmax=1, vmin=-2)
    plt.tight_layout()
    plt.title("Heat map of covariance matrix")
    plt.show()


    print('\n **********************************')
    pearson_corr=df.corr(method='pearson')
    print(f' Pearson Correlation coefficients Matrix is: \n {pearson_corr.to_string()}')
    sns.heatmap(pearson_corr, cmap="crest", vmax=1, vmin=-2)
    plt.tight_layout()
    plt.title("Heat map of Pearson Correlation coefficients Matrix")
    plt.show()



def data_analysis():
    df=pd.read_csv(r"C:\Users\saich\Documents\books\DA\Diamonds Prices2022.csv")
    print(df.info())
    missing_values=df.isnull().sum()
    print(f'missing values in the data set\n{missing_values}')
    df.drop(["Unnamed: 0"],axis=1,inplace=True)

    df,df1,target=one_hot(df)
    regression_analysis.regression_models(df, target)
    feature_selection(df,target)
    correlation(df)
    #print(df.head())
    df_scaled = standardize(df, df1)
    df_scaled.drop(['price'],axis=1,inplace=True)

    return df_scaled,target













