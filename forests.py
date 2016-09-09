import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

# Note: file is comma-delimited
X = np.genfromtxt("data/trainX.txt",delimiter=',')
Y = np.genfromtxt("data/trainY.txt",delimiter=',')
# also load features of the test data (to be predicted)
Xe1 = np.genfromtxt("data/devX.txt",delimiter=',')
Ye1 = np.genfromtxt("data/devY.txt",delimiter=',')
print X.shape
print Y.shape

nBag = 10;

m,n = X.shape
classifiers = [ None ] * nBag # Allocate space for learners

for i in range(nBag):
    Xi, Yi = ml.bootstrapData(X,Y);
    classifiers[i] = ml.dtree.treeRegress(Xi, Yi , maxDepth=20, minParent=1024,nFeatures=60) # Train a model on data Xi, Yi

#training errors
trainingErrors = np.zeros( nBag ) # Allocate space for predictions from each model
for i in range(nBag):
    temp = np.sqrt(classifiers[i].mse(X,Y)) # Apply each classifier
    trainingErrors[i] = temp;
# Make overall prediction by majority vote
#tE = np.mean(trainingErrors, axis=)

# test on data Xtest
predict = np.zeros( nBag ) # Allocate space for predictions from each model
for i in range(nBag):
    temp = np.sqrt(classifiers[i].mse(Xe1, Ye1)) # Apply each classifier
    predict[i] = temp;
# Make overall prediction by majority vote
#p = np.mean(predict, axis=1)
#p = p[:,0]

Xt = np.mean(trainingErrors)
Xe = np.mean(predict)

print Xt;
print Xe;

