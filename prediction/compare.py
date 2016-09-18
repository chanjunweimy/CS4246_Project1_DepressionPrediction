from math import sqrt
from sklearn import gaussian_process
from sklearn import cross_validation
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

X_train, y_train, X_dev, y_dev = read_train_dev_files("trainX.txt", "devX.txt", "trainY.txt", "devY.txt")

classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(2),
    "Linear SVM": SVC(kernel="linear"),
    "RBF SVM": SVC(gamma=2, C=1),
    "Decision Tree": DecisionTreeClassifier(min_samples_split=1024, max_features=60, max_depth=20),
    "Random Forest": RandomForestClassifier(n_estimators=10, min_samples_split=1024, max_features=60, max_depth=20),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "GP abs_exp": gaussian_process.GaussianProcess(corr='absolute_exponential'),
    "GP squared_exp": gaussian_process.GaussianProcess(corr='squared_exponential'),
    #General Expo doesn't work with the data: 'Exception: Length of theta must be 2 or 525'
    "GP cubic": gaussian_process.GaussianProcess(corr='cubic'),
    "GP linear": gaussian_process.GaussianProcess(corr='linear'),
    "Bagging with DTRegg": BaggingRegressor(DecisionTreeRegressor(min_samples_split=1024, max_depth=20, max_features=60),
                                            n_estimators=10, max_samples=1.0, max_features=1.0)}

models_rmse = {}
for name, model in classifiers.items():
    model.fit(X_train[:], y_train[:])
    rmse_train = sqrt(mean_squared_error(y_train, model.predict(X_train)))
    rmse_predict = sqrt(mean_squared_error(y_dev, model.predict(X_dev)))
    models_rmse[name] = [rmse_train, rmse_predict]
    print(name)
    print("\tT:" + str(rmse_train)+"\n\tP:"+str(rmse_predict))

plot_bar(models_rmse)
#plot_all_Y()