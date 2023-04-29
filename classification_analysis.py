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
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from yellowbrick.classifier import ROCAUC


def analysis_plot(y_pred,y_test,y_prob,model):
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
    dt=DecisionTreeClassifier()
    dt.fit(x_train,y_train)
    y_pred=dt.predict(x_test)
    y_prob=dt.predict_proba(x_test)
    analysis_plot(y_pred,y_test,y_prob,"Decision Tree")
    visualizer = ROCAUC(dt)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()


def logistic_regression(x_train,x_test,y_train,y_test):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train.values,y_train.values)
    y_pred = lr.predict(x_test.values)
    y_prob = lr.predict_proba(x_test.values)
    analysis_plot(y_pred, y_test, y_prob, "Logistic regression")
    visualizer = ROCAUC(lr)
    visualizer.fit(x_train.values, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test.values, y_test)  # Evaluate the model on the test data
    visualizer.show()


def knn(x_train,x_test,y_train,y_test):
    knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_prob = knn.predict_proba(x_test)
    analysis_plot(y_pred, y_test, y_prob, "KNN")
    visualizer = ROCAUC(knn)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()

def svm(x_train,x_test,y_train,y_test):
    svm = SVC(kernel='rbf',probability=True)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    y_prob = svm.predict_proba(x_test)
    analysis_plot(y_pred, y_test, y_prob, "SVM")
    visualizer = ROCAUC(svm)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()

def naive_bayes(x_train,x_test,y_train,y_test):
    nb=GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    y_prob = nb.predict_proba(x_test)
    analysis_plot(y_pred, y_test, y_prob, "Naive Bayes")
    visualizer = ROCAUC(nb)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()

def random_forest(x_train,x_test,y_train,y_test):
    rf=RandomForestClassifier()
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    y_prob=rf.predict_proba(x_test)
    analysis_plot(y_pred, y_test, y_prob, "Random Forest ")
    visualizer = ROCAUC(rf)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show()

def neural_net(x_train,x_test,y_train,y_test):
    print(x_train.shape)
    model = Sequential([
        Flatten(input_shape=(21,)),
        # dense layer 1
        Dense(512, activation='sigmoid'),
        # dense layer 2
        Dense(256, activation='sigmoid'),
        #dense layer 3
        Dense(128, activation='sigmoid'),
        # output layer
        Dense(6, activation='sigmoid'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100,
              batch_size=200,
              validation_split=0.2)
    results = model.evaluate(x_test, y_test, verbose=0)
    print('test loss, test acc:', results)


def train_model(df,target):
    x_train, x_test, y_train, y_test = train_test_split(df,target, test_size=0.2, random_state=42)
    # print("************************************************************")
    # #decision_tree(x_train, x_test, y_train, y_test)
    # print("************************************************************")
    logistic_regression(x_train, x_test, y_train, y_test)
    # #print("************************************************************")
    # #knn(x_train, x_test, y_train, y_test)
    # print("************************************************************")
    # #svm(x_train, x_test, y_train, y_test)
    # print("************************************************************")
    # naive_bayes(x_train, x_test, y_train, y_test)
    # print("************************************************************")
    # random_forest(x_train, x_test, y_train, y_test)
    print("************************************************************")
    #neural_net(x_train, x_test, y_train, y_test)
    print("************************************************************")