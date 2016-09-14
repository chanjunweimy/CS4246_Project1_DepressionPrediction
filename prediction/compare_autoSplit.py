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
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings

warnings.filterwarnings("ignore")

X = []
y = []
n_x = 0
n_y = 0
n = 0
n_feats = 0

with open('trainX.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    X = [ [ float(eaVal) for eaVal in row] for row in reader]
    # safety to check every row
    n_feats = len(X[0])
    for x in X:
        if n_feats != len(x):
            print('Warning, some x has different number of features!!')
            sys.exit(1)
n_x = len(X)

with open('trainY.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    y = [ int(row[0]) for row in reader ]
n_y = len(y)

if n_x != n_y:
    print('Error, n_x != n_y')
    sys.exit(1)
n = n_x

print("Data has " + str(n_feats) + " features and " + str(n) + " data points." )

# some baselines
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Gaussian Process",
         "Bagging with DTRegg"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=10, max_features=60, max_depth=20),
    AdaBoostClassifier(),
    GaussianNB(),
    gaussian_process.GaussianProcess(),
    BaggingRegressor(DecisionTreeRegressor(min_samples_split=1024, max_depth=20, max_features=60, random_state=0), n_estimators=10, max_samples=1.0, max_features=1.0, random_state=0)]
classifierPair = zip(names, classifiers)

for eaPair in classifierPair:
    scores = cross_validation.cross_val_score(eaPair[1], X[:], y[:], n_jobs=-1, cv=10, scoring='mean_squared_error')
    print(eaPair[0] +":" + str(sqrt(-1*np.mean(scores))))

