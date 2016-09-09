import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

# Note: file is comma-delimited
X = np.genfromtxt("data/emotion_dcapswoz_mfcc.txt",delimiter=',')
Y = np.genfromtxt("data/emotion_dcapswoz_y.txt",delimiter=',')
# also load features of the test data (to be predicted)
print X.shape
print Y.shape

nBag = 101;
learners = np.array([2, 5, 10, 20, 50]);

classifiers = [ None ] * nBag # Allocate space for learners

errT = np.zeros((len(learners),))

nFolds = 5
errX = np.zeros((len(learners),nFolds))
for iFold in range(nFolds):
    [Xti,Xvi,Yti,Yvi] = ml.crossValidate(X,Y,nFolds,iFold)
    for i in range(nBag):
        Xi, Yi = ml.bootstrapData(Xti,Yti);
        classifiers[i] = ml.dtree.treeRegress(Xi, Yi , maxDepth=20, minParent=1024,nFeatures=60) # Train a model on data Xi, Yi
    for i in range(len(learners)):
        learnerNum = learners[i];
        predict = np.zeros( (learnerNum) ) # Allocate space for predictions from each model
        for j in range(learnerNum):
            predict[j] = classifiers[j].mse(Xvi,Yvi) # Apply each classifier
        errX[i, iFold] = np.mean(predict);

errX = np.mean(errX, axis=1) 
print errX.shape;

for i in range(nBag):
    Xi, Yi = ml.bootstrapData(X,Y);
    classifiers[i] = ml.dtree.treeRegress(Xi, Yi , maxDepth=20, minParent=1024,nFeatures=60) # Train a model on data Xi, Yi
for i in range(len(learners)):
    learnerNum = learners[i];
    predict = np.zeros( (learnerNum) ) # Allocate space for predictions from each model
    for j in range(learnerNum):
        predict[j] = classifiers[j].mse(X,Y) # Apply each classifier
    errT[i] = np.mean(predict);
    
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.semilogy(learners,errT,'r+-', # training error (from P1)
#learners,errV,'g-', # validation error (from P1)
learners,errX,'m+-', # cross-validation estimate of validation error
linewidth=2);   
plt.axis([1,50,0,0.2]);
plt.show();

for i in range(len(learners)):
    print learners[i];
    print errT[i];
    print errX[i];