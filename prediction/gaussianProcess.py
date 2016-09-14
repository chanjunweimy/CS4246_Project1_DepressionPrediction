import numpy as np
from sklearn import gaussian_process
import csv
import sys
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from math import sqrt
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore")

def getX(fileName):
    X = []
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        X = [ [ float(eaVal) for eaVal in row] for row in reader]
        # safety to check every row
        n_feats = len(X[0])
        for x in X:
            if n_feats != len(x):
                print('Warning, some x has different number of features!!')
                sys.exit(1)
    return X, n_feats, len(X)

X_train, n_feats_train, k_x_train = getX('trainX.txt')
X_dev, n_feats_dev, k_x_dev = getX('devX.txt')

# some sanity checks on n_feats
if n_feats_train != n_feats_dev:
    print('Error n_feats in train and dev. They are not equal.')
    sys.exit(1)
n_feats = n_feats_train

def getY(filename):
    y = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        y = [ int(row[0]) for row in reader ]
    return y, len(y)

y_train, k_y_train = getY('trainY.txt')
y_dev, k_y_dev = getY('devY.txt')

# sanity checks for k_train
if k_x_train != k_y_train:
    print('Error, train is of different size')
    sys.exit(1)
k_train = k_x_train
if k_x_dev != k_y_dev:
    print('Error, dev is of different size')
    sys.exit(1)
k_dev = k_x_dev

print("Data has " + str(n_feats) + " features and " + str(k_train) + " training points and " + str(k_dev) + " dev points." )

# some baselines
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Gaussian Process"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(n_estimators=10, max_features=None, min_samples_split=1024, max_depth=20),
    AdaBoostClassifier(),
    GaussianNB(),
    gaussian_process.GaussianProcess()]
classifierPair = zip(names, classifiers)

for eaPair in classifierPair:
    #scores = cross_validation.cross_val_score(eaPair[1], X[:], y[:], n_jobs=-1, scoring='mean_squared_error')
    model = eaPair[1].fit(X_train[:], y_train[:])
    y_pred = model.predict(X_dev)
    rmse = sqrt(mean_squared_error(y_dev, y_pred))
    print(eaPair[0] +":" + str(rmse))
