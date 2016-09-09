## IMPORTS #####################################################################
import math
import numpy as np

from numpy import atleast_2d as twod
from numpy import asmatrix as mat
from numpy import asmatrix as arr

from .utils import toIndex


################################################################################
## Base (abstract) "classify" class and associated functions ###################
################################################################################

class classifier:

  def __init__(self, *args, **kwargs):
    """
    Constructor for base class for several different classifiers. 
    This class implements methods that generalize to different classifiers.
    Optional arguments X,Y,... call train(X,Y,...) to initialize the model
    """
    self.classes = []
    # TODO: if Y!=None init classes from data? (leave to train?)
    if len(args) or len(kwargs):
        return self.train(*args, **kwargs)


  def __call__(self, *args, **kwargs):
    """
    This method provides syntatic sugar for prediction; it simply calls "predict".
    """ 
    return self.predict(*args, **kwargs)

#Removed; just leave classes as a property
#  def getClasses(self):
#    """
#    Return the list of class identifiers for the classifier
#    """
#    return self.classes

  def predict(self, X):
    """
    This is an abstract predict method that must exist in order to
    implement certain classifier methods.
    Input:
      X :  MxN numpy data matrix; each row corresponds to one data point
    Output:
      Y :  Mx1 vector of predicted class for each data point
    The default implementation uses predictSoft and converts to the most likely class.
    """
    #return np.argmax( self.predictSoft(X) , axis=1 )
    idx = np.argmax( self.predictSoft(X) , axis=1 )      # find most likely class (index)
    #return self.classes[idx]                        # convert to saved class values
    return np.asarray([[self.classes[i]] for i in np.ravel(idx)])   # convert to saved class values


  def predictSoft(self,X):
    """
    This is an abstract prediction method that must exist in order to
    use many "soft" classification methods.
    Input:
      X :  MxN numpy data matrix; each row corresponds to one data point
    Output:
      P :  MxC numpy class probability matrix; each column corresponds to the
          "probability" (or confidence) that the datum is in that class
    """
    raise NotImplementedError

  ####################################################
  # Standard loss f'n definitions for classifiers    #
  ####################################################
  def err(self, X, Y):
    """
    This method computes the error rate on test data.  

    Parameters
    ---------
    X : M x N numpy array 
      M = number of data points; N = number of features. 
    Y : M x 1 numpy array    
      Array of classes (targets) corresponding to the data points in X.
    """
    Y    = arr( Y )
    Yhat = arr( self.predict(X) )
    return np.mean(Yhat.reshape(Y.shape) != Y)


  def nll(self, X, Y):
    """
    This method computes the (average) negative log-likelihood
    of the soft predictions (intepreted as probabilities):
      (1/M) \sum_i log Pr[ y^{(i)} | f, x^{(i)} ]

    Parameters
    ---------
    X : M x N numpy array 
      M = number of data points; N = number of features. 
    Y : M x 1 numpy array   
      Array of target values (classes) for each datum in X
    """
    M,N = X.shape
    P = arr( self.predictSoft(X) )
    P /= np.sum(P, axis=1, keepdims=True)       # normalize to sum to one
    Y = toIndex(Y, self.classes)
    return - np.mean( np.log( P[ range(M), Y ] ) ) # evaluate



  def auc(self, X, Y):
    """
    This method computes the area under the roc curve on the given test data.
    This method only works on binary classifiers. 

    Parameters
    ---------
    X : M x N numpy array 
      M = number of data points; N = number of features. 
    Y : M x 1 numpy array 
      Array of classes (targets) corresponding to the data points in X.
    """
    if len(self.classes) != 2:
      raise ValueError('This method can only supports binary classification ')

    try:                  # compute 'response' (soft binary classification score)
      soft = self.predictSoft(X)[:,1]  # p(class = 2nd)
    except (AttributeError, IndexError):  # or we can use 'hard' binary prediction if soft is unavailable
      soft = self.predict(X)

    n,d = twod(soft).shape             # ensure soft is the correct shape
    soft = soft.flatten() if n==1 else soft.T.flatten()

    indices = np.argsort(soft)         # sort data by score value
    Y = Y[indices]
    sorted_soft = soft[indices]

    # compute rank (averaged for ties) of sorted data
    dif = np.hstack( ([True],np.diff(sorted_soft),[True]) )
    r1  = np.argwhere(dif).flatten()
    r2  = r1[0:-1] + 0.5*(r1[1:]-r1[0:-1]) + 0.5
    rnk = r2[np.cumsum(dif[:-1])-1]

    # number of true negatives and positives
    n0,n1 = sum(Y == self.classes[0]), sum(Y == self.classes[1])

    if n0 == 0 or n1 == 0:
      raise ValueError('Data of both class values not found')

    # compute AUC using Mann-Whitney U statistic
    result = (np.sum(rnk[Y == self.classes[1]]) - n1 * (n1 + 1) / 2) / n1 / n0
    return result


  def confusion(self, X, Y):
    """
    This method estimates the confusion matrix (Y x Y_hat) from test data.
    
    Parameters
    ---------
    X : M x N numpy array 
      M = number of data points; N = number of features. 
    Y : M x 1 numpy array 
      Array of classes (targets) corresponding to the data points in X.
    """
    Y_hat = self.predict(X)
    num_classes = len(self.classes)
    indices = toIndex(Y, self.classes) + num_classes * (toIndex(Y_hat, self.classes) - 1)
    C = np.histogram(indices, np.asarray(range(1, num_classes**2 + 2)))[0]
    C = np.reshape(C, (num_classes, num_classes))
    return np.transpose(C)


  def roc(self, X, Y):
    """
    This method computes the "receiver operating characteristic" curve on
    test data.  This method is only defined for binary classifiers. Refer 
    to the auc doc string for descriptions of X and Y. Method returns
    (fpr, tpr, tnr), where
      fpr = false positive rate (1xN numpy vector)
      tpr = true positive rate (1xN numpy vector)
      tnr = true negative rate (1xN numpy vector)
    Plot fpr vs. tpr to see the ROC curve. 
    Plot tpr vs. tnr to see the sensitivity/specificity curve.
    """
    if len(self.classes) > 2:
      raise ValueError('This method can only supports binary classification ')

    try:                  # compute 'response' (soft binary classification score)
      soft = self.predictSoft(X)[:,1]  # p(class = 2nd)
    except (AttributeError, IndexError):
      soft = self.predict(X)        # or we can use 'hard' binary prediction if soft is unavailable
    n,d = twod(soft).shape

    if n == 1:
      soft = soft.flatten()
    else:
      soft = soft.T.flatten()

    # number of true negatives and positives
    n0 = np.sum(Y == self.classes[0])
    n1 = np.sum(Y == self.classes[1])

    if n0 == 0 or n1 == 0:
      raise ValueError('Data of both class values not found')

    # sort data by score value
    sorted_soft = np.sort(soft)
    indices = np.argsort(soft)

    Y = Y[indices]

    # compute false positives and true positive rates
    tpr = np.divide(np.cumsum(Y[::-1] == self.classes[1]), n1)
    fpr = np.divide(np.cumsum(Y[::-1] == self.classes[0]), n0)
    tnr = np.divide(np.cumsum(Y == self.classes[0]), n0)[::-1]

    # find ties in the sorting score
    same = np.append(np.asarray(sorted_soft[0:-1] == sorted_soft[1:]), 0)
    tpr = np.append([0], tpr[np.logical_not(same)])
    fpr = np.append([0], fpr[np.logical_not(same)])
    tnr = np.append([1], tnr[np.logical_not(same)])
    return [tpr, fpr, tnr]



################################################################################
## REGRESS #####################################################################
################################################################################


class regressor:

  def __init__(self, *args, **kwargs):
    """
    Constructor for base class for several different regression learners. 
    This class implements methods that generalize to different regressors.
    """
    if len(args) or len(kwargs):
        return self.train(*args, **kwargs)



  def __call__(self, *args, **kwargs):
    """
    This method provides syntatic sugar for prediction; it simply calls "predict".
    """ 
    return self.predict(*args, **kwargs)



  ####################################################
  # Standard loss f'n definitions for regressors     #
  ####################################################
  def mae(self, X, Y):
    """
    This method computes the mean absolute error,
      (1/M) \sum_i | f(x^{(i)}) - y^{(i)} |
    of a regression model f(.) on test data X and Y. 

    Args:
      X = M x N numpy array that contains M data points with N features
      Y = M x 1 numpy array of target values for each data point
    """
    Yhat = self.predict(X)
    return np.mean(np.absolute(Y - Yhat.reshape(Y.shape)), axis=0)


  def mse(self, X, Y):
    """
    This method computes the mean squared error,
      (1/M) \sum_i ( f(x^{(i)}) - y^{(i)} )^2 
    of a regression model f(.) on test data X and Y. 

    Args:
      X = M x N numpy array that contains M data points with N features
      Y = M x 1 numpy array of target values for each data point
    """
    Yhat = self.predict(X)
    #print Y.shape, Yhat.shape
    return np.mean( (Y - Yhat.reshape(Y.shape))**2 , axis=0)


  def rmse(self, X, Y):
    """
    This method computes the root mean squared error, 
      sqrt( f.mse(X,Y) )
    of a regression model f(.) on test data X and Y. 

    Args:
      X = M x N numpy array that contains M data points with N features
      Y = M x 1 numpy array of target values for each data point
    """
    return np.sqrt(self.mse(X, Y))



################################################################################
################################################################################
################################################################################
