import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import regression_analysis

def one_hot(df):
    le = LabelEncoder()
    for var in ['cut','color','clarity']:
        df[var] = le.fit_transform(df[var])
    return df

def standardize(df):
    scalar = StandardScaler()
    df1=df[['cut','color','clarity']]
    df.drop(['cut', 'color', 'clarity'], axis=1, inplace=True)
    values = scalar.fit_transform(df)
    df_scalar = pd.DataFrame(values, columns=df.columns)
    df_scalar.boxplot()
    plt.show()
    df = pd.concat([df_scalar, df1], axis=1)
    return df

def remove_outliers(df,target):
    for col in df.columns:
        upper_bound = df[col].quantile(0.99)
        lower_bound = df[col].quantile(0.01)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    df=pd.concat([df,target],axis=1)
    df.dropna(inplace=True,axis=0)
    target=df['price']
    df.drop(['price'],axis=1,inplace=True)
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
    #x_scaled=StandardScaler().fit_transform(df)
    pca = PCA()
    X_pca = pca.fit_transform(df)
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
    cols=df.columns
    u, s, vh = np.linalg.svd(df, full_matrices=False)
    print("Singular values:\n", s)
    pct_variance_to_retain = 0.95
    total_variance = sum(s ** 2)
    variance_explained = [(i / total_variance) for i in (s ** 2)]

    cumulative_variance_explained = np.cumsum(variance_explained)
    num_singular_values_to_keep = np.argmax(cumulative_variance_explained >= pct_variance_to_retain) + 1
    selected_cols = cols[:num_singular_values_to_keep]
    print(f'selected columns\n{selected_cols}')


def feature_selection_classification(df,target):
    print('*********RandomForest Classification Analysis************')

    x_train,x_test,y_train,y_test=train_test_split(df,target,test_size=0.2,shuffle=True,random_state=5525)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_
    feature_names = df.columns
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance',                                                                                          ascending=False)
    print(feature_importances.head(10))


    print('*********PCA Classification Analysis************')
    #x_scaled=StandardScaler().fit_transform(df)
    pca = PCA()
    X_pca = pca.fit_transform(df)
    e_values=np.round(pca.explained_variance_,2)
    variance_ratio=pca.explained_variance_ratio_
    print(f'Variance explained by all  principal components = {sum(variance_ratio* 100):.2f}')
    print(np.cumsum(variance_ratio * 100))

    plt.figure()
    plt.plot(np.cumsum(variance_ratio*100))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.title('Elbow plot for PCA-Classification')
    plt.show()

    print('\n ***************SVD Classification*******************')
    cols = df.columns
    u, s, vh = np.linalg.svd(df, full_matrices=False)
    print("Singular values:\n", s)
    pct_variance_to_retain = 0.95
    total_variance = sum(s ** 2)
    variance_explained = [(i / total_variance) for i in (s ** 2)]

    cumulative_variance_explained = np.cumsum(variance_explained)
    num_singular_values_to_keep = np.argmax(cumulative_variance_explained >= pct_variance_to_retain) + 1
    selected_cols = cols[:num_singular_values_to_keep]
    print(f'selected columns\n{selected_cols}')



def correlation(df,met):
    print('\n **********************************')
    print(f'Covariance Matrix is: \n {df.cov().to_string()}')
    cov_heat = df.cov()
    sns.heatmap(cov_heat, annot=True, fmt="0.02f",vmax=1, vmin=-2)
    plt.tight_layout()
    plt.title(f"-Heat map of covariance matrix")
    plt.show()


    print('\n **********************************')
    pearson_corr=df.corr(method='pearson')
    print(f'Pearson Correlation coefficients Matrix is: \n {pearson_corr.to_string()}')
    sns.heatmap(pearson_corr, annot=True, fmt="0.02f",vmax=1, vmin=-2)
    plt.tight_layout()
    plt.title(f"Heat map of Pearson Correlation coefficients Matrix")
    plt.show()




def data_analysis(df,target):
    feature_selection(df,target)
    return df,target













