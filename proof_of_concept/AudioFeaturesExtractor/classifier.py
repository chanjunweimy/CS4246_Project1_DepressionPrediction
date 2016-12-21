from math import sqrt, isnan
from sklearn import gaussian_process as gp
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import cross_val_score#, ShuffleSplit
from skfeature.function.statistical_based import CFS
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.similarity_based import reliefF
from inputs import read_train_dev_files_with_binary
from plotting import plot_bar, plot_all_Y, plot_f1
import numpy as np
import sys
import GPy.kern as kern
import GPy.models as models
import time
from math import sqrt,ceil
import GPy
import GPyOpt
import matplotlib.mlab as mlab
import math
import matplotlib.pyplot as plt

x_train_file_name = "data/splitted/X/MFCC/trainX.txt"
x_dev_file_name = "data/splitted/X/MFCC/devX.txt"
y_train_file_name = "data/splitted/y/trainY.txt"
y_dev_file_name = "data/splitted/y/devY.txt"

y_bin_train_file_name = "data/splitted/bin_Y/trainY.txt"
y_bin_dev_file_name = "data/splitted/bin_Y/devY.txt"

if len(sys.argv) == 5:
    x_train_file_name = sys.argv[1]
    x_dev_file_name = sys.argv[2]
    y_train_file_name = sys.argv[3]
    y_dev_file_name = sys.argv[4]
elif len(sys.argv) == 3:
    x_train_file_name = sys.argv[1]
    x_dev_file_name = sys.argv[2]
elif len(sys.argv) == 2:    
     x_dev_file_name = sys.argv[1]
#print len(sys.argv)
#print sys.argv

X_train, y_train, X_dev, y_dev, y_bin_train, y_bin_dev = read_train_dev_files_with_binary(x_train_file_name, x_dev_file_name, y_train_file_name, y_dev_file_name, y_bin_train_file_name, y_bin_dev_file_name)
n_feats = len(X_train[0])

nfold = 3
def fit_svc_val(x):
    x = np.atleast_2d(np.exp(x))
    fs = np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        fs[i] = 0
        for n in range(nfold):
            idx = np.array(range(X_train.shape[0]))
            idx_valid = np.logical_and(idx>=X_train.shape[0]/nfold*n, idx<X_train.shape[0]/nfold*(n+1))
            idx_train = np.logical_not(idx_valid)
            svc = SVC(C=x[i,0] ,gamma=x[i,1])
            #svc.fit(X_train[idx_train],y_bin_train[idx_train])
            f1, performance = getClassifierPerformanceOfXAndY(svc, "SVC-BO", "MFCC", 
            X_train[idx_train], y_bin_train[idx_train], X_train[idx_valid], y_bin_train[idx_valid])
            fs[i] += 1 - f1[1]
            #fs[i] += np.sqrt(np.square(svr.predict(X_train[idx_valid])-Y_train[idx_valid]).mean())
        fs[i] *= 1./nfold
    return fs
    


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
        
def mean(a):
    return sum(a) / len(a)
    
def getEnsemblesPerformances(previous_layer, current_layer, models_f1, models_performances): 
    X_temp_train = []
    X_temp_dev = []
    for name, featSelectionMode, model in previous_layer:            
        model.fit(X_train, y_bin_train)
        X_temp_train.append(model.predict(X_train))
        X_temp_dev.append(model.predict(X_dev))
    
    X_new_train = map(list, zip(*X_temp_train)) # Transpose list of lists
    X_new_dev = map(list, zip(*X_temp_dev)) # Transpose list of lists
    
    for name, featSelectionMode, model in current_layer:            
        f1, performance = getClassifierPerformance(model, name, "MFCC", X_new_train, y_bin_train, X_new_dev)
        models_f1.append(f1)
        models_performances.append(performance)
    return models_f1, models_performances
        
def getClassifieresPerformances(classifiers, models_f1, models_performances): 
    for name, featSelectionMode, model in classifiers:            
        f1, performance = getClassifierPerformance(model, name, "MFCC", X_train, y_bin_train, X_dev)
        models_f1.append(f1)
        models_performances.append(performance)
    return models_f1, models_performances
    
def getClassifieresPerformancesByDefinedX(classifiers, eaMode, models_f1, models_performances, X, Y, Xstar): 
    for name, featSelectionMode, model in classifiers:
        f1, performance = getClassifierPerformance(model, name, eaMode, X, Y, Xstar)
        models_f1.append(f1)
        models_performances.append(performance)
    return models_f1, models_performances
    
def getClassifierPerformance(model, name, eaMode, X, Y, X_star):
    model.fit(X, Y)
    #pt_f1, pt_precision, pt_recall, pt_accuracy = classifyForF1(model, X, 1)
    pp_f1, pp_precision, pp_recall, pp_accuracy = classifyForF1(model, X_star, 1)
    #nt_f1, nt_precision, nt_recall, nt_accuracy = classifyForF1(model, X, 0)
    np_f1, np_precision, np_recall, np_accuracy = classifyForF1(model, X_star, 0)
    f1 = [name + '(' + eaMode + ')', pp_f1, np_f1]
    performance = [name + '(' + eaMode + ')', pp_f1, pp_precision, pp_recall, pp_accuracy, np_f1, np_precision, np_recall, np_accuracy]
    return f1, performance

def getClassifierPerformanceOfXAndY(model, name, eaMode, X, Y, X_star, Y_star):
    model.fit(X, Y)
    #pt_f1, pt_precision, pt_recall, pt_accuracy = classifyForF1WithY(model, X, y_bin_train, 1)
    pp_f1, pp_precision, pp_recall, pp_accuracy = classifyForF1WithY(model, X_star, Y_star, 1)
    #nt_f1, nt_precision, nt_recall, nt_accuracy = classifyForF1WithY(model, X, y_bin_train, 0)
    np_f1, np_precision, np_recall, np_accuracy = classifyForF1WithY(model, X_star, Y_star,0)
    f1 = [name + '(' + eaMode + ')', pp_f1, np_f1]
    performance = [name + '(' + eaMode + ')', pp_f1, pp_precision, pp_recall, pp_accuracy, np_f1, np_precision, np_recall, np_accuracy]
    return f1, performance

def classifyForF1WithY(classifier, X, Y, positive_bit):
    #depressionClassifer = gp.GaussianProcessClassifier(kernel=gp.kernels.DotProduct())
    #classifier.fit(newTrainX, y_bin_train)
    classifiedResult = classifier.predict(X)
    #classifiedResultProba = classifier.predict_proba(newDevX)
    #print classifiedResult
    #print classifiedResultProba

    #define 1 as +ve 0 as negative
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for i in range(len(Y)):
        actual = Y[i]
        predicted = classifiedResult[i]
        
        if actual == positive_bit and predicted == actual:
            tp = tp + 1.0
        elif actual == positive_bit and predicted != actual:
            fp = fp + 1.0
        elif actual == 1 - positive_bit and predicted == actual:
            tn = tn + 1.0
        elif actual == 1 - positive_bit and predicted != actual:
            fn = fn + 1.0
        else:
            print 'got bug'
      

    f1 = -0.01
    precision = -0.01
    recall = -0.01
    accuracy = -0.01
      
      
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    
    if len(y_bin_dev) != 0:
        accuracy = (tp + tn) / len(Y)
    
    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    
    if tp != 0 and precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    #print 'precision:' + str(precision)
    #print 'recall:' + str(recall)
    #print 'accuracy:' + str(accuracy)
    #print 'f1:' + str(f1)
    return f1, precision, recall, accuracy
    
def classifyForF1(classifier, X, positive_bit):
    return classifyForF1WithY(classifier, X, y_bin_dev, positive_bit)
    
def addRelatedWork(models_f1, models_performances):
    f1 = ['DepAudioNet', 0.52, 0.70]
    performance = ['DepAudioNet', 0.52, 0.35, 1.00, '-', 0.70, 1.00, 0.54, '-']
    models_f1.append(f1)
    models_performances.append(performance)
    return models_f1, models_performances
    
def printPerformances(models_performances):
    for performance in models_performances:
        name = performance[0]
        pp_f1 = performance[1]
        pp_precision = performance[2]
        pp_recall = performance[3]
        pp_accuracy = performance[4]
        np_f1 = performance[5]
        np_precision = performance[6]
        np_recall = performance[7]
        np_accuracy = performance[8]
        print name
        print ('\tF1: ' + str(pp_f1) + '(' + str(np_f1) + 
        '),Precision: ' + str(pp_precision) + '(' + str(np_precision) + 
        '),Recall: ' + str(pp_recall) + '(' + str(np_recall) + 
        '),Accuracy: '  + str(pp_accuracy) + '(' + str(np_accuracy) + ')') 
        


#mode = "MFCC"
#bitVec = bitVecs[mode]


#depressionRegressor = gp.GaussianProcessRegressor(kernel=gp.kernels.DotProduct())
#depressionRegressor.fit(X_train[:,bitVec], y_train)
#phq8 = depressionRegressor.predict(X_dev[:,bitVec])
#rmse_train = sqrt(mean_squared_error(y_train, depressionRegressor.predict(X_train[:,bitVec])))
#rmse_predict = sqrt(mean_squared_error(y_dev, phq8))
#print rmse_train
#print rmse_predict

domain       =[{'name': 'C',      'type': 'continuous', 'domain': (0.,7.)},
               {'name': 'gamma',  'type': 'continuous', 'domain': (-12.,-2.)}]
opt = GPyOpt.methods.BayesianOptimization(f = fit_svc_val,            # function to optimize       
                                         domain = domain,         # box-constrains of the problem
                                         #acquisition_type ='LCB',       # LCB acquisition
                                         acquisition_type ='EI',     # http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_reference_manual.ipynb  
                                         acquisition_weight = 0.1)   # Exploration exploitation
# it may take a few seconds
opt.run_optimization(max_iter=50)
# opt.plot_convergence()
x_best = np.exp(opt.X[np.argmin(opt.Y)])

domain       =[{'name': 'C',      'type': 'continuous', 'domain': (0.,7.)},
               {'name': 'gamma',  'type': 'continuous', 'domain': (-12.,-2.)}]
opt = GPyOpt.methods.BayesianOptimization(f = fit_svc_val,            # function to optimize       
                                         domain = domain,         # box-constrains of the problem
                                         #acquisition_type ='LCB',       # LCB acquisition
                                         acquisition_type ='MPI',     # http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_reference_manual.ipynb  
                                         acquisition_weight = 0.1)   # Exploration exploitation
# it may take a few seconds
opt.run_optimization(max_iter=50)
# opt.plot_convergence()
x_best_2 = np.exp(opt.X[np.argmin(opt.Y)])


classifiers = [("KNN", None, KNeighborsClassifier(2)),
               ("Linear SVM", None, SVC(kernel="linear")),
               ("RBF SVM", None, SVC(gamma=2, C=1)),
               ("OPT SVM_EI", None, SVC(C=x_best[0], gamma=x_best[1])),
               ("OPT SVM_MPI", None, SVC(C=x_best_2[0], gamma=x_best_2[1])),
               ("DT", None, DecisionTreeClassifier(min_samples_split=1024, max_depth=20)),
               ("RF", None, RandomForestClassifier(n_estimators=10, min_samples_split=1024,
                                                         max_depth=20)),
               ("AB", None, AdaBoostClassifier(random_state=13370)),
               #("GP ARD", ["MFCC"], gp.GaussianProcessClassifier(kernel=ard_kernel(sigma=1.2, length_scale=np.array([1]*1)))),
               ("GP-DP", ["MFCC"], gp.GaussianProcessClassifier(kernel=gp.kernels.DotProduct()))
               # output the confidence level and the predictive variance for the dot product (the only one that we keep in the end)
               # GP beats SVM in our experiment (qualitative advantages)
               # only keep RBF, dot product and matern on the chart
               # add a paragraph 'Processed Data'
               #1) generate the dataset with 526 features
               #2) the predictive variance and predictive mean (best and worst) of some vectors from the dot product.
               
]
#classify(X_train[:,bitVec], X_dev[:,bitVec])

ensembles = [("KNN_ENS", None, KNeighborsClassifier(2)),
               ("Linear SVM_ENS", None, SVC(kernel="linear")),
               ("RBF SVM_ENS", None, SVC(gamma=2, C=1)),
               ("DT_ENS", None, DecisionTreeClassifier(min_samples_split=1024, max_depth=20)),
               ("RF_ENS", None, RandomForestClassifier(n_estimators=10, min_samples_split=1024,
                                                         max_depth=20)),
               ("AB_ENS", None, AdaBoostClassifier(random_state=13370)),
               #("GP ARD", ["MFCC"], gp.GaussianProcessClassifier(kernel=ard_kernel(sigma=1.2, length_scale=np.array([1]*1)))),
               ("GP-DP_ENS", ["MFCC"], gp.GaussianProcessClassifier(kernel=gp.kernels.DotProduct()))
               # output the confidence level and the predictive variance for the dot product (the only one that we keep in the end)
               # GP beats SVM in our experiment (qualitative advantages)
               # only keep RBF, dot product and matern on the chart
               # add a paragraph 'Processed Data'
               #1) generate the dataset with 526 features
               #2) the predictive variance and predictive mean (best and worst) of some vectors from the dot product.
               
]


models_f1 = []
models_performances = []
models_f1, models_performances = getClassifieresPerformances(classifiers, models_f1, models_performances)
models_f1, models_performances = getEnsemblesPerformances(classifiers, ensembles, models_f1, models_performances)
#models_f1, models_performances = getClassifieresPerformancesByDefinedX(classifiers, 'predict', models_f1, models_performances, newTrainX, y_bin_train, newDevX)
models_f1, models_performances = addRelatedWork(models_f1, models_performances)
models_f1=sorted(models_f1, key=lambda l: l[1])
models_performances=sorted(models_performances, key=lambda l: l[1])

plot_f1(models_f1)
printPerformances(models_performances)