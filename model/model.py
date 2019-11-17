from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
#Import svm model
import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def model(file_path):
    df = pd.read_csv(file_path)

    X = df.drop('label', axis=1)
    y = df['label']  # setting up testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
    sm = SMOTE(random_state=27, ratio=1.0)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    y_train = pd.DataFrame(y_train, columns=['label'])
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Fitting Kernel SVM to the Training set
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train.values.ravel())
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)



model(file_path='/home/asingh/workspace/mist_play/mist_play/data/labeled_data.csv')
