from math import sqrt
from sklearn import gaussian_process as gp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score#, ShuffleSplit
from skfeature.function.statistical_based import CFS
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.similarity_based import reliefF
from inputs import read_train_dev_files
from plotting import plot_bar, plot_all_Y
import numpy as np
import sys
from ARD_kernel import ard_kernel
import GPy.kern as kern
import GPy.models as models
import time
from math import sqrt,ceil

x_train_file_name = "data/splitted/X/urop/trainX.txt"
x_dev_file_name = "data/splitted/X/urop/devX.txt"
y_train_file_name = "data/splitted/y/trainY.txt"
y_dev_file_name = "data/splitted/y/devY.txt"

if len(sys.argv) == 5:
    x_train_file_name = sys.argv[1]
    x_dev_file_name = sys.argv[2]
    y_train_file_name = sys.argv[3]
    y_dev_file_name = sys.argv[4]
elif len(sys.argv) == 3:
    x_train_file_name = sys.argv[1]
    x_dev_file_name = sys.argv[2]

X_train, y_train, X_dev, y_dev = read_train_dev_files(x_train_file_name, x_dev_file_name, y_train_file_name, y_dev_file_name)
n_feats = len(X_train[0])

# We perform feature selection first
numFeatsFn = lambda n: int(ceil(sqrt(n_feats)))
def reliefPostProc(X, y):
    scores = reliefF.reliefF(X,y)
    indexes = range(0, len(scores))
    pairedScores = zip(scores, indexes)
    pairedScores = sorted(pairedScores, reverse=True)
    return np.array([ eaPair[1] for eaPair in pairedScores][:numFeatsFn(n_feats)])

def baselineProc(X,y):
    return range(0,n_feats)

def ml(X, y):
    ml = gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=0.5))
    scores = cross_val_score(ml, X, y, cv=5, n_jobs=-1, scoring='mean_squared_error') #problem
    #scores = 0
    return sqrt(-1*np.mean(scores))

def convertToBitVec(featSel):
    def wrapper(X, y):
        feats = featSel(X,y)
        bitVec = [False] * n_feats
        for eaF in feats:
            bitVec[eaF] = True
        bitVec = np.array(bitVec)
        return len(feats),bitVec
    return wrapper


# CIFE: index of selected features, F[1] is the most important feature
# CFS: index of selected features
# RELIEF: index of selected features, F[1] is the most important feature
featSelectionFns = {
    "All": convertToBitVec(baselineProc),
    "Relief": convertToBitVec(reliefPostProc),
    "CIFE": convertToBitVec(CIFE.cife),
    "CFS": convertToBitVec(CFS.cfs)
}
timeTaken = []
bitVecs = {}

for featSelName, featSel in featSelectionFns.iteritems():
    start = time.clock()    
    numFeats,bitVec = featSel(X_train,y_train)
    timeTaken = time.clock() - start
    score = ml(X_train[:,bitVec], y_train) #problem
    bitVecs[featSelName] = bitVec
    print(featSelName+ "," + str(numFeats) + ": " + str(score) + " in "+ str(timeTaken) + "seconds")


class RBF_ARD_WRAPPER:
    def __init__(self, kernel_ardIn):
        self.kernel_ard = kernel_ardIn
    def fit(self, X_trainIn, y_trainIn):
        y_train_l = y_trainIn.reshape((y_trainIn.shape[0], 1))
        self.m = models.GPRegression(X_trainIn, y_train_l, self.kernel_ard)
        self.m.constrain_positive('')
        self.m.optimize_restarts(num_restarts=10)
        self.m.randomize()
        self.m.optimize()
    def predict(self, X):
        return self.m.predict(X)[0]


classifiers = [("Nearest Neighbors", None, KNeighborsClassifier(2)),
               ("Linear SVM", None, SVC(kernel="linear")),
               #("RBF SVM", None, SVC(gamma=2, C=1)),
               ("Decision Tree", None, DecisionTreeClassifier(min_samples_split=1024, max_depth=20)),
               ("Random Forest", None, RandomForestClassifier(n_estimators=10, min_samples_split=1024,
                                                         max_depth=20)),
               ("AdaBoost", None, AdaBoostClassifier()),
               ("Naive Bayes", None, GaussianNB()),
               ("Bagging with DTRegg", ["All"], BaggingRegressor(DecisionTreeRegressor(min_samples_split=1024,
                                                                              max_depth=20))),
               ("GP isotropic RBF", None, gp.GaussianProcessRegressor(kernel=gp.kernels.RBF())),
               ("GP anisotropic RBF", ["All"], gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(length_scale=np.array([1]*n_feats)))),
               ("GP ARD", ["All"], gp.GaussianProcessRegressor(kernel=ard_kernel(sigma=1.2, length_scale=np.array([1]*n_feats)))),
               ("GP isotropic matern nu=0.5", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=0.5))),
               ("GP isotropic matern nu=1.5", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=1.5))),
               ("GP isotropic matern nu=2.5", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=2.5))),
# bad performance
               ("GP dot product", ["CFS", "CIFE", "All"], gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())),
#  3-th leading minor not positive definite
#    ("GP exp sine squared", gp.GaussianProcessRegressor(kernel=gp.kernels.ExpSineSquared())),
               ("GP rational quadratic", None, gp.GaussianProcessRegressor(kernel=gp.kernels.RationalQuadratic())),
               ("GP white kernel", None, gp.GaussianProcessRegressor(kernel=gp.kernels.WhiteKernel())),
               ("GP abs_exp", None, gp.GaussianProcess(corr='absolute_exponential')),
               ("GP squared_exp", ["All"], gp.GaussianProcess(corr='squared_exponential')),
               ("GP cubic", None, gp.GaussianProcess(corr='cubic')),
               ("GP linear", None, gp.GaussianProcess(corr='linear')),
               ("GP RBF ARD", ["All"], RBF_ARD_WRAPPER(kern.RBF(input_dim=n_feats, variance=1., lengthscale=np.array([1]*n_feats), ARD=True)))]


models_rmse = []
for name, featSelectionMode, model in classifiers:
    modes = featSelectionMode
    if featSelectionMode==None:
        modes = featSelectionFns.keys()
    rmses = []
    for eaMode in modes:
        bitVec = bitVecs[eaMode]
        model.fit(X_train[:,bitVec], y_train[:])
        rmse_train = sqrt(mean_squared_error(y_train, model.predict(X_train[:,bitVec])))
        rmse_predict = sqrt(mean_squared_error(y_dev, model.predict(X_dev[:,bitVec])))
        rmses.append([name + '('+eaMode+')', rmse_train, rmse_predict])
        print(name + '('+eaMode+')')
        print("\tT:" + str(rmse_train)+"\n\tP:"+str(rmse_predict))
    rmses=sorted(rmses, key=lambda l: l[2])
    models_rmse.append(rmses[0])
plot_bar(models_rmse)
#plot_all_Y()
