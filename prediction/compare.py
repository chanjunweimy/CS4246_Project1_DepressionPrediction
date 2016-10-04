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
from inputs import read_train_dev_files
from plotting import plot_bar, plot_all_Y
import numpy as np
import sys
from ARD_kernel import ard_kernel
import GPy.kern as kern
import GPy.models as models


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

classifiers = [("Nearest Neighbors", KNeighborsClassifier(2)),
               ("Linear SVM", SVC(kernel="linear")),
               ("RBF SVM", SVC(gamma=2, C=1)),
               ("Decision Tree", DecisionTreeClassifier(min_samples_split=1024, max_features=60, max_depth=20)),
               ("Random Forest", RandomForestClassifier(n_estimators=10, min_samples_split=1024,
                                                        max_features=60, max_depth=20)),
               ("AdaBoost", AdaBoostClassifier()),
               ("Naive Bayes", GaussianNB()),
               ("Bagging with DTRegg", BaggingRegressor(DecisionTreeRegressor(min_samples_split=1024,
                                                                              max_depth=20, max_features=60))),
               ("GP isotropic RBF", gp.GaussianProcessRegressor(kernel=gp.kernels.RBF())),
               ("GP anisotropic RBF", gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(length_scale=np.array([1]*n_feats)))),
               ("GP ARD", gp.GaussianProcessRegressor(kernel=ard_kernel(sigma=1.2, length_scale=np.array([1]*n_feats)))),
               ("GP isotropic matern nu=0.5", gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=0.5))),
               ("GP isotropic matern nu=1.5", gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=1.5))),
               ("GP isotropic matern nu=2.5", gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=2.5))),
               ("GP dot product", gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())),
#  3-th leading minor not positive definite
#    ("GP exp sine squared", gp.GaussianProcessRegressor(kernel=gp.kernels.ExpSineSquared())),
               ("GP rational quadratic", gp.GaussianProcessRegressor(kernel=gp.kernels.RationalQuadratic())),
               ("GP white kernel", gp.GaussianProcessRegressor(kernel=gp.kernels.WhiteKernel())),
               ("GP abs_exp", gp.GaussianProcess(corr='absolute_exponential')),
               ("GP squared_exp", gp.GaussianProcess(corr='squared_exponential')),
               ("GP cubic", gp.GaussianProcess(corr='cubic')),
               ("GP linear", gp.GaussianProcess(corr='linear'))]


models_rmse = []
for name, model in classifiers:
    model.fit(X_train[:], y_train[:])
    rmse_train = sqrt(mean_squared_error(y_train, model.predict(X_train)))
    rmse_predict = sqrt(mean_squared_error(y_dev, model.predict(X_dev)))
    models_rmse.append([name, rmse_train, rmse_predict])
    print(name)
    print("\tT:" + str(rmse_train)+"\n\tP:"+str(rmse_predict))

def add_rbf_ard(Xtrain, ytrain, rmse):
    X_train = np.array(Xtrain)
    y_train = np.array(ytrain)
    y_train = y_train.reshape((y_train.shape[0], 1))

    kernel_ard = kern.RBF(input_dim=n_feats, variance=1., lengthscale=np.array([1]*n_feats), ARD=True)
    m = models.GPRegression(X_train, y_train, kernel_ard)

    m.constrain_positive('')
    m.optimize_restarts(num_restarts=10)
    m.randomize()
    m.optimize()

    train_mean, train_var = m.predict(X_train)
    dev_mean, dev_var = m.predict(X_dev)

    rmse_train = sqrt(mean_squared_error(y_train, train_mean))
    rmse_dev = sqrt(mean_squared_error(y_dev, dev_mean))

    rmse.append(["GP RBF ARD", rmse_train, rmse_dev])

add_rbf_ard(X_train, y_train, models_rmse)

plot_bar(models_rmse)
#plot_all_Y()
