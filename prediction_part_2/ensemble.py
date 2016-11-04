from math import sqrt
from sklearn import gaussian_process as gp
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import cross_val_score#, ShuffleSplit
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
import GPy
import GPyOpt
import matplotlib.mlab as mlab
import math
import matplotlib.pyplot as plt

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
    #scores = cross_val_score(ml, X, y, cv=5, n_jobs=-1, scoring='mean_squared_error') #problem
    scores = 0
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

mfccIntVec = [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def returnMask(intVec):
    def wrapper(X,y):
        mask = intVec
        mask = np.array(mask).astype('bool')
        return np.sum(mask),mask
    return wrapper

# CIFE: index of selected features, F[1] is the most important feature
# CFS: index of selected features
# RELIEF: index of selected features, F[1] is the most important feature
featSelectionFns = {
    "All": convertToBitVec(baselineProc),
    "Relief": convertToBitVec(reliefPostProc),
    "CIFE": convertToBitVec(CIFE.cife),
    "CFS": convertToBitVec(CFS.cfs),
    "MFCC": returnMask(mfccIntVec)
}
timeTaken = []
bitVecs = {}

print 'ok1'

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
        
def createModelBasedOnName(name):
    if name == "KNN":
        return KNeighborsRegressor(2)
    elif name == "Linear SVR":
        return SVR(kernel="linear")
    elif name == "RBF SVR":
        return SVR(gamma=2, C=1)
    elif name == "DT":
        return DecisionTreeRegressor(min_samples_split=1024, max_depth=20)
    elif name == "RF":
        return RandomForestRegressor(n_estimators=10, min_samples_split=1024, max_depth=20)
    elif name == "AB":
        return AdaBoostRegressor(random_state=13370)
    elif name == "GP-DP":
        return gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())
    
    return None
               

class AverageRacialEnsemble:
    def __init__(self, learners):
        self.learners = learners
    
    #todo: bayesian optimization when fitting
    def fit(self, X, Y):
        for learner in self.learners:
            learner['model'].fit(X,Y)
    
    def predict(self, X):
        scores = []
        for learner in self.learners:
            predictions = learner['model'].predict(X)
            tempY = []
            for prediction in predictions:
                tempY.append(prediction * learner['weight'])
            scores.append(tempY)
        #print scores
        npScores = np.array(scores)
        y = np.sum(npScores, axis = 0)
        #print y
        return y

class GaussianEnsembleExtended:
    def __init__(self, learners):
        self.learners = []
        self.modelsSpecified = learners
        self.gp = gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())
    
    def fit(self, X, Y):    
        newX = []
        for i in range(len(X)):
            newX.append([])
        
        for modelSpecified in self.modelsSpecified:
            modes = modelSpecified['modes']
            if modes == None:
                modes = featSelectionFns.keys()
            for eaMode in modes:
                bitVec = bitVecs[eaMode]
                model = createModelBasedOnName(modelSpecified['name'])
                if model == None:
                    print modelSpecified['name']
                    continue
                model.fit(X[:,bitVec], Y[:])
                tempYs = model.predict(X[:,bitVec])
                for i in range(len(tempYs)):
                    tempY = tempYs[i]
                    newX[i].append(tempY)
                learner = {"mode": eaMode, "model": model}
                self.learners.append(learner)
        
        self.gp.fit(newX,Y)
    
    def predict(self, X):
        newX = []
        for i in range(len(X)):
            newX.append([])
        
        for learner in self.learners:
            mode = learner['mode']
            bitVec = bitVecs[mode]
            model = learner['model']
            tempYs = model.predict(X[:,bitVec])
            for i in range(len(tempYs)):
                tempY = tempYs[i]
                newX[i].append(tempY)
        y = self.gp.predict(newX)
        return y    
        
        
class GaussianEnsemble:
    def __init__(self, learners):
        self.learners = learners
        self.gp = gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())
    
    def fit(self, X, Y):      
        for learner in self.learners:
            learner['model'].fit(X,Y)
        newX = []
        for eachX in X:
            newXRow = []
            for learner in self.learners:
                tempY = learner['model'].predict(eachX)
                newXRow.append(tempY[0])
            newX.append(newXRow)
        self.gp.fit(newX,Y)
    
    def predict(self, X):
        newX = []
        for eachX in X:
            newXRow = []
            for learner in self.learners:
                tempY = learner['model'].predict(eachX)
                newXRow.append(tempY[0])
            newX.append(newXRow)
        y = self.gp.predict(newX)
        return y        
        
class GaussianAverageEnsemble:
    def __init__(self, learners):
        self.gp = gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())
        self.avg = AverageRacialEnsemble(learners)
    
    def fit(self, X, Y):
        self.avg.fit(X,Y)
        avg_y = self.avg.predict(X)
        newX = []
        for eachY in avg_y:
            newRow = []
            newRow.append(eachY)
            newX.append(newRow)
        self.gp.fit(newX,Y)
    
    def predict(self, X):
        avg_y = self.avg.predict(X)
        newX = []
        for eachY in avg_y:
            newRow = []
            newRow.append(eachY)
            newX.append(newRow)
        y = self.gp.predict(newX)
        return y        
        
class BasesianOptimisationRacialEnsemble:    
    learners = []
    weights = []
    
    @staticmethod
    def setLearners(learners):
        learners = learners
    
    @staticmethod
    def fit(X, Y):
        for learner in learners:
            learner['model'].fit(X,Y)
        
        for i in range(len(learners)):
            weights.append(1.0 / len(learners))
            
    @staticmethod
    def ensembleFunction(w0, w1, w2, w3, w4, w5, w6):
        return w0 * learners[0]['model'].predict(X)
    
    @staticmethod
    def predict(X):
        scores = []
        for learner in self.learners:
            predictions = learner['model'].predict(X)
            tempY = []
            for prediction in predictions:
                tempY.append(prediction * learner['weight'])
            scores.append(tempY)
        #print scores
        npScores = np.array(scores)
        y = np.sum(npScores, axis = 0)
        #print y
        return y
        
def trainModels(regressors, models_rmse): 
    for name, featSelectionMode, model in regressors:
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
    return models_rmse

regressors = [("KNN", None, KNeighborsRegressor(2)),
               ("Linear SVR", None, SVR(kernel="linear")),
               ("RBF SVR", None, SVR(gamma=2, C=1)),
               ("DT", None, DecisionTreeRegressor(min_samples_split=1024, max_depth=20)),
               ("RF", None, RandomForestRegressor(n_estimators=10, min_samples_split=1024,
                                                         max_depth=20)),
               ("AB", None, AdaBoostRegressor(random_state=13370)),
               #("Naive Bayes", None, GaussianNB()),
               #("Bagging with DTRegg", ["All"], BaggingRegressor(DecisionTreeRegressor(min_samples_split=1024,
                #                                                              max_depth=20))),
               #("GP isotropic RBF", None, gp.GaussianProcessRegressor(kernel=gp.kernels.RBF())),
               #("GP anisotropic RBF", ["All"], gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(length_scale=np.array([1]*n_feats)))),
               #("GP ARD", ["All"], gp.GaussianProcessRegressor(kernel=ard_kernel(sigma=1.2, length_scale=np.array([1]*n_feats)))),
               #("GP isotropic matern nu=0.5", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=0.5))),
               #("GP isotropic matern nu=1.5", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=1.5))),
               #("GP Isotropic Matern", None, gp.GaussianProcessRegressor(kernel=gp.kernels.Matern(nu=2.5))),
# bad performance
               ("GP-DP", ["MFCC","All","CIFE","CFS"], gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())),
               # output the confidence level and the predictive variance for the dot product (the only one that we keep in the end)
               # GP beats SVM in our experiment (qualitative advantages)
               # only keep RBF, dot product and matern on the chart
               # add a paragraph 'Processed Data'
               #1) generate the dataset with 526 features
               #2) the predictive variance and predictive mean (best and worst) of some vectors from the dot product.

#  3-th leading minor not positive definite
#    ("GP exp sine squared", gp.GaussianProcessRegressor(kernel=gp.kernels.ExpSineSquared())),
               #("GP rational quadratic", None, gp.GaussianProcessRegressor(kernel=gp.kernels.RationalQuadratic())),
               #("GP white kernel", None, gp.GaussianProcessRegressor(kernel=gp.kernels.WhiteKernel())),
               #("GP abs_exp", None, gp.GaussianProcess(corr='absolute_exponential')),
               #("GP squared_exp", ["All"], gp.GaussianProcess(corr='squared_exponential')),
               #("GP cubic", None, gp.GaussianProcess(corr='cubic')),
               #("GP linear", None, gp.GaussianProcess(corr='linear')),
               #("GP RBF ARD", ["All"], RBF_ARD_WRAPPER(kern.RBF(input_dim=n_feats, variance=1., lengthscale=np.array([1]*n_feats), ARD=True)))]
               
]

RacialEnsembles = []

learners = []
averageWeight = 1.0 / len(regressors)
weights = [averageWeight, averageWeight, averageWeight, averageWeight, averageWeight, averageWeight, averageWeight]
i = 0
for name, featSelectionMode, model in regressors:
    learner = {'weight': weights[i], 'model': model, 'modes': featSelectionMode, 'name': name}
    learners.append(learner)
    i += 1

RacialEnsembles.append(("ARE", ["MFCC","All","CIFE","CFS"], AverageRacialEnsemble(learners)))
RacialEnsembles.append(("GE", ["MFCC","All","CIFE","CFS"], GaussianEnsemble(learners)))
RacialEnsembles.append(("GAE", ["MFCC","All","CIFE","CFS"], GaussianAverageEnsemble(learners)))
#RacialEnsembles.append(("GEE", ["All"], GaussianEnsembleExtended(learners))) #GEE doesn't work well


models_rmse = []
models_rmse = trainModels(regressors, models_rmse)
models_rmse = trainModels(RacialEnsembles, models_rmse)

# Give some general prior distributions for model parameters
# m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
# m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
# m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
# _=m.plot()

#print("AverageRacialEnsemble(MFCC)")
#print("\tT:" + str(rmse_train)+"\n\tP:"+str(rmse_predict))
plot_bar(models_rmse)
#plot_all_Y()



