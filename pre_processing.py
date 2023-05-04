import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import regression_analysis

def one_hot(df):
    le = LabelEncoder()
    for var in ['cut','color','clarity']:
        df[var] = le.fit_transform(df[var])
    print(f'one-hot:\n{df.head()}')
    return df

def standardize(df):
    scalar = StandardScaler()
    print(df.columns)
    df1=df[['cut','color','clarity']]
    df.drop(['cut', 'color', 'clarity'], axis=1, inplace=True)
    values = scalar.fit_transform(df)
    df_scalar = pd.DataFrame(values, columns=df.columns)
    df_scalar.boxplot()
    plt.show()
    df = pd.concat([df_scalar, df1], axis=1)
    return df

def remove_outliers(df,target):
    columns = df.columns
    print(f'col:{columns}')
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
    outliers = lof.fit_predict(df[columns])
    outlier_indices = np.where(outliers == -1)[0]
    df = df.drop(df.index[outlier_indices])
    target = target.drop(target.index[outlier_indices])
    df.boxplot()
    plt.show()
    return df,target

def feature_selection(df,target):
    print('*********RandomForest Analysis************')

    x_train,x_test,y_train,y_test=train_test_split(df,target,test_size=0.2,shuffle=True,random_state=5525)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_
    feature_names = df.columns
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance',                                                                                          ascending=False)
    print(feature_importances.head(10))


    print('*********PCA Analysis************')
    df1=df.drop(['cut','color','clarity'],axis=1)
    x_scaled=StandardScaler().fit_transform(df1)
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

    print('\n ***************SVD*******************')



    svd = TruncatedSVD(n_components=7)
    X_svd = svd.fit_transform(df)
    print(f'svd:{X_svd.shape}\n{pd.DataFrame(X_svd).head()}')

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



def data_analysis(df,target):


    df=one_hot(df)
    df= standardize(df)
    df,target=remove_outliers(df,target)
    #feature_selection(df,target)
    #print(df.head())
    df.drop(['x','y','table','depth'],axis=1,inplace=True)
    correlation(df)
    regression_analysis.regression_models(df,target)

    return df,target













