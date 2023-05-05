from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from yellowbrick.classifier import ROCAUC
from yellowbrick.cluster import KElbowVisualizer




def analysis_plot(y_pred,y_test,model):
    conf_matrix= confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="0.02f")
    plt.title(f"{model} Confusion matrix")
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.show()
    report = classification_report(y_test, y_pred)
    print(f'{model} Accuracy score={accuracy_score(y_test,y_pred)*100}')
    print(f'{model} Classification report:\n{report}')



def decision_tree(x_train,x_test,y_train,y_test):
    dt=DecisionTreeClassifier(random_state=42)
    dt.fit(x_train,y_train)
    y_pred=dt.predict(x_test)
    y_prob=dt.predict_proba(x_test)
    analysis_plot(y_pred,y_test,"Decision Tree")
    visualizer = ROCAUC(dt)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()


def logistic_regression(x_train,x_test,y_train,y_test):
    lr = LogisticRegression(multi_class="multinomial",max_iter=1000,random_state=42)
    lr.fit(x_train.values,y_train.values)
    y_pred = lr.predict(x_test.values)
    y_prob = lr.predict_proba(x_test.values)
    analysis_plot(y_pred, y_test, "Logistic regression")
    visualizer = ROCAUC(lr)
    visualizer.fit(x_train.values, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test.values, y_test)  # Evaluate the model on the test data
    visualizer.show()


def knn(x_train,x_test,y_train,y_test):
    knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_prob = knn.predict_proba(x_test)
    analysis_plot(y_pred, y_test, "KNN")
    visualizer = ROCAUC(knn)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()

def svm(x_train,x_test,y_train,y_test):
    svm = SVC(kernel='rbf',probability=True,random_state=42)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    y_prob = svm.predict_proba(x_test)
    analysis_plot(y_pred, y_test, "SVM")
    visualizer = ROCAUC(svm)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()

def naive_bayes(x_train,x_test,y_train,y_test):
    nb=GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    y_prob = nb.predict_proba(x_test)
    analysis_plot(y_pred, y_test, "Naive Bayes")
    visualizer = ROCAUC(nb)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()

def random_forest(x_train,x_test,y_train,y_test):
    rf=RandomForestClassifier(random_state=42)
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    analysis_plot(y_pred, y_test, "Random Forest ")
    visualizer = ROCAUC(rf)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()

def neural_net(df,target):
    X = df
    y = target
    y = pd.get_dummies(y, prefix='price')  # convert categorical variable to one-hot encoding
    X = np.array(X)
    y = np.array(y)
    tf.random.set_seed(42)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100,
              batch_size=200,
              validation_split=0.2,verbose=1)

    results = model.evaluate(x_test, y_test, verbose=0)
    print('test loss, test acc:', results)

def kmeans(x_train,x_test,y_train,y_test):
    kmeans = KMeans(n_clusters=3,algorithm='auto',max_iter=2000, random_state=42)
    kmeans.fit(x_train)
    y_pred = kmeans.predict(x_test)
    le=LabelEncoder()
    y_test=le.fit_transform(y_test)
    analysis_plot(y_pred, y_test, "K-Means")



def dbscan(x_train,x_test,y_train,y_test):
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(x_train)
    y_pred = dbscan.predict(x_test)
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    analysis_plot(y_pred, y_test, "DBSCAN ")

def apriori(x_train,x_test,y_train,y_test):
    frequent_itemsets = apriori(x_train, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print(rules)



def train_model(df,target):
    print(f'{df.head()}\n\n{target.head()}\n\n{df.shape,target.shape}')
    x_train, x_test, y_train, y_test = train_test_split(df,target, test_size=0.2, random_state=42)
    print("**************************DT*********************************")
    decision_tree(x_train, x_test, y_train, y_test)
    print("**************************LR*********************************")
    logistic_regression(x_train, x_test, y_train, y_test)
    print("**************************KNN*********************************")
    knn(x_train, x_test, y_train, y_test)
    print("**************************SVM********************************")
    #svm(x_train, x_test, y_train, y_test)
    print("*************************Naive_Bayes*********************************")
    naive_bayes(x_train, x_test, y_train, y_test)
    print("***************************RF********************************")
    random_forest(x_train, x_test, y_train, y_test)
    print("***************************NeuralNet********************************")
    #neural_net(df,target)
    print("****************************K-Means********************************")
    kmeans(x_train, x_test, y_train, y_test)
    print("****************************DBSCAN********************************")
    dbscan(x_train, x_test, y_train, y_test)
    print("****************************Apriori********************************")
    #apriori(x_train, x_test, y_train, y_test)