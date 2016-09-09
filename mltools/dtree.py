import numpy as np

from .base import classifier
from .base import regressor
from .utils import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat


################################################################################
## TREEREGRESS #################################################################
################################################################################


class treeRegress(regressor):

    def __init__(self, *args, **kwargs):
        """
        Constructor for treeRegress (decision tree regression model)

        Parameters: see "train" function; calls "train" if arguments passed

        Properties (internal use only)
           L,R : indices of left & right child nodes in the tree
           F,T : feature index & threshold for decision (left/right) at this node
                 for leaf nodes, T[n] holds the prediction for leaf node n
        """
        self.L = arr([0])           # indices of left children
        self.R = arr([0])           # indices of right children
        self.F = arr([0])           # feature to split on (-1 = leaf = predict)
        self.T = arr([0])           # threshold to split on (prediction value if leaf)
   
        if len(args) or len(kwargs):     # if we were given optional arguments,
            self.train(*args, **kwargs)    #  just pass them through to "train"
 
    
    def __repr__(self):
        to_return = 'Decision Tree Regressor\n'
        if len(self.T) > 8:
            to_return += 'Thresholds: {}'.format(
                '[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
                .format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
        else:
            to_return = self.__printTree(0,'  ')
        return to_return

    def __printTree(self,node,indent):
        to_return = ''
        if (self.F[node] == -1):
            to_return += indent+'Predict {}\n'.format(self.T[node])
        else:
            to_return += indent+'if x[{:d}] < {:f}:\n'.format(int(self.F[node]),self.T[node])
            to_return += self.__printTree(self.L[node],indent+'  ')
            to_return += indent+'else:\n'
            to_return += self.__printTree(self.R[node],indent+'  ')
        return to_return

    __str__ = __repr__


#    def __str__(self):
#        to_return = 'Decision Tree Regressor \nThresholds: {}'.format(
#            str(self.T) if len(self.T) < 4 else 
#            '[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
#            .format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
#        return to_return


## CORE METHODS ################################################################


    def train(self, X, Y, minParent=2, maxDepth=np.inf, minScore=-1, nFeatures=None):
        """
        Train a decision-tree regressor model

        Parameters
        ----------
        X : M x N numpy array of M data points with N features each
        Y : numpy array of shape (M,) that contains the target values for each data point
        minParent : (int)   Minimum number of data required to split a node. 
        minScore  : (float) Minimum value of score improvement to split a node.
        maxDepth  : (int)   Maximum depth of the decision tree. 
        nFeatures : (int)   Number of available features for splitting at each node.
        """
        n,d = mat(X).shape
        nFeatures = min(nFeatures if nFeatures else d, d)

        sz = min(2 * n, 2**(maxDepth + 1))   # pre-allocate storage for tree:
        L, R, F, T = np.zeros((sz,)), np.zeros((sz,)), np.zeros((sz,)), np.zeros((sz,))

        L, R, F, T, last = self.__dectree_train(X, Y, L, R, F, T, 0, 0, minParent, maxDepth, minScore, nFeatures)

        self.L = L[0:last]                              # store returned data into object
        self.R = R[0:last]                              
        self.F = F[0:last]
        self.T = T[0:last]



    def predict(self, X):
        """
        Make predictions on the data in X

        Parameters
        ----------
        X : M x N numpy array containing M data points of N features each
        """
        return self.__dectree_test(X, self.L, self.R, self.F, self.T, 0)


    
## HELPERS #####################################################################


    def __dectree_train(self, X, Y, L, R, F, T, next, depth, minParent, maxDepth, minScore, nFeatures):
        """
        This is a recursive helper method that recusively trains the decision tree. Used in:
            train

        TODO:
            compare for numerical tolerance
        """
        n,d = mat(X).shape

        # check leaf conditions...
        if n < minParent or depth >= maxDepth or np.var(Y) < minScore:
            assert n != 0, ('TreeRegress.__dectree_train: tried to create size zero node')
            return self.__output_leaf(Y, n, L, R, F, T, next)

        best_val = np.inf
        best_feat = -1
        try_feat = np.random.permutation(d)

        # ...otherwise, search over (allowed) features
        for i_feat in try_feat[0:nFeatures]:
            dsorted = arr(np.sort(X[:,i_feat].T)).ravel()                       # sort data...
            pi = np.argsort(X[:,i_feat].T)                                      # ...get sorted indices...
            tsorted = Y[pi].ravel()                                             # ...and sort targets by feature ID
            can_split = np.append(arr(dsorted[:-1] != dsorted[1:]), 0)          # which indices are valid split points?

            if not np.any(can_split):          # no way to split on this feature?
                continue

            # find min weighted variance among split points
            val,idx = self.__min_weighted_var(tsorted, can_split, n)

            # save best feature and split point found so far
            if val < best_val:
                best_val = val
                best_feat = i_feat
                best_thresh = (dsorted[idx] + dsorted[idx + 1]) / 2

        # if no split possible, output leaf (prediction) node
        if best_feat == -1:         
            return self.__output_leaf(Y, n, L, R, F, T, next)

        # split data on feature i_feat, value (tsorted[idx] + tsorted[idx + 1]) / 2
        F[next] = best_feat
        T[next] = best_thresh
        go_left = X[:,F[next]] < T[next]
        my_idx = next
        next += 1

        # recur left
        L[my_idx] = next    
        L,R,F,T,next = self.__dectree_train(X[go_left,:], Y[go_left], L, R, F, T, 
            next, depth + 1, minParent, maxDepth, minScore, nFeatures)

        # recur right
        R[my_idx] = next    
        L,R,F,T,next = self.__dectree_train(X[np.logical_not(go_left),:], Y[np.logical_not(go_left)], L, R, F, T, 
            next, depth + 1, minParent, maxDepth, minScore, nFeatures)

        return (L,R,F,T,next)


    def __dectree_test(self, X, L, R, F, T, pos):
        """
        This is a recursive helper method that finds leaf nodes
        in the decision tree for prediction. Used in:
            predict
        """
        M,N = X.shape
        y_hat = np.zeros((M,1))
        
        if F[pos] == -1:
            y_hat[:] = T[pos]
        else:
            go_left = X[:,F[pos]] < T[pos]  # which data should follow left split?
            y_hat[go_left]  = self.__dectree_test(X[go_left,:],  L, R, F, T, L[pos])
            go_right = np.logical_not(go_left)  # other data go right:
            y_hat[go_right] = self.__dectree_test(X[go_right,:], L, R, F, T, R[pos])

        return y_hat


    def __output_leaf(self, Y, n, L, R, F, T, next):
        """
        This is a helper method that handles leaf node termination
        conditions. Used in:
            __dectree_train
        """
        F[next] = -1
        T[next] = np.mean(Y)        
        next += 1
        return (L,R,F,T,next)


    def __min_weighted_var(self, tsorted, can_split, n):
        """
        This is a helper method that finds the minimum weighted variance
        among all split points. Used in:
            __dectree_train
        """
        # compute mean up to and past position j (for j = 0..n)
        y_cum_to = np.cumsum(tsorted, axis=0)
        y_cum_pa = y_cum_to[-1] - y_cum_to
        #mean_to = y_cum_to / arr(range(1, n + 1))       
        #mean_pa = y_cum_pa / arr(list(range(n - 1, 0, -1)) + [1])
        count_to = np.arange(1.0,n+1)
        count_pa = np.arange(1.0*n - 1, -1, -1)
        count_pa[-1] = 1.0
        mean_to = y_cum_to / count_to; #np.arange(1.0, n + 1)       
        mean_pa = y_cum_pa / count_pa; #np.arange(1.0*n - 1, -1, -1)

        # compute variance up to, and past position j (for j = 0..n)
        y2_cum_to = np.cumsum(np.power(tsorted, 2), axis=0)
        y2_cum_pa = y2_cum_to[-1] - y2_cum_to
        #var_to = (y2_cum_to - 2 * mean_to * y_cum_to + list(range(1, n + 1)) * np.power(mean_to, 2)) / list(range(1, n + 1))
        #var_pa = (y2_cum_pa - 2 * mean_pa * y_cum_pa + list(range(n - 1, -1, -1)) * np.power(mean_pa, 2)) / arr(list(range(n - 1, 0, -1)) + [1])
        var_to = (y2_cum_to - 2 * mean_to * y_cum_to + count_to * np.power(mean_to, 2)) / count_to
        var_pa = (y2_cum_pa - 2 * mean_pa * y_cum_pa + count_pa * np.power(mean_pa, 2)) / count_pa

        # find minimum weighted variance among all split points
        #weighted_variance = np.arange(1.0, n + 1)/n * var_to + np.arange(1.0*n - 1, -1, -1)/n * var_pa
        weighted_variance = count_to/n * var_to + count_pa/n * var_pa
        weighted_variance[-1] = np.inf
        weighted_variance[can_split==0] = np.inf   # find only splittable points
        #idx = np.nanargmin((weighted_variance + 1) / (can_split + 1e-100))      # find only splittable points
        idx = np.nanargmin(weighted_variance)      # use nan version to ignore any nan values
        val = weighted_variance[idx]
        #val = np.nanmin((weighted_variance + 1) / (can_split + 1e-100))         # nan versions of min functions must be used to ignore nans

        return (val,idx)


################################################################################
################################################################################
################################################################################


################################################################################
## TREECLASSIFY ################################################################
################################################################################


class treeClassify(classifier):

    def __init__(self, *args, **kwargs):
        """
        Constructor for TreeClassifier (decision tree classifier).

        Parameters: see the "train" function; calls "train" if arguments passed

        Properties:
          classes : list of identifiers for each class
          TODO
        """
        self.classes = []
        self.L = np.asarray([0])         # index of left child
        self.R = np.asarray([0])         # index of right child
        self.F = np.asarray([0])         # feature to split on
        self.T = np.asarray([0])         # threshold value for feature

        if len(args) or len(kwargs):     # if we were given optional arguments,
            self.train(*args, **kwargs)    #  just pass them through to "train"

    
#    def X__repr__(self):
#        to_return = 'Decision Tree Classifier; {} features\nThresholds: {}'.format(
#            len(self.classes), str(self.T) if len(self.T) < 4 else 
#            '[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
#             .format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
#        return to_return

    def __repr__(self):
        to_return = 'Decision Tree Classifier\n'
        if len(self.T) > 8:
            to_return += 'Thresholds: {}'.format(
                '[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
                .format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
        else:
            to_return = self.__printTree(0,'  ')
        return to_return

    def __printTree(self,node,indent):
        to_return = ''
        if (self.F[node] == -1):
            to_return += indent+'Predict {}\n'.format(self.T[node])
        else:
            to_return += indent+'if x[{:d}] < {:f}:\n'.format(int(self.F[node]),self.T[node])
            to_return += self.__printTree(self.L[node],indent+'  ')
            to_return += indent+'else:\n'
            to_return += self.__printTree(self.R[node],indent+'  ')
        return to_return

    __str__ = __repr__

#    def __str__(self):
#        to_return = 'Decision Tree Classifier; {} features\nThresholds: {}'.format(
#            len(self.classes), str(self.T) if len(self.T) < 4 else 
#            '[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
#             .format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
#        return to_return


## CORE METHODS ################################################################


    def train(self, X, Y, minParent=2, maxDepth=np.inf, nFeatures=None):
        """
        Trains a random forest classification tree.

        Parameters
        ----------
        X : M x N numpy array of M data points with N features each.
        Y : M x 1 numpy array containing class labels for each data point in X. 
        minParent : (int) The minimum number of data required to split a node. 
        maxDepth  : (int) The maximum depth of the decision tree. 
        nFeatures : (int) The number of available features for splitting at each node.
        """
        n,d = arr(X).shape
        nFeatures = d if nFeatures is None else min(nFeatures,d)
        minScore = -1

        self.classes = list(np.unique(Y)) if len(self.classes) == 0 else self.classes
        Y = toIndex(Y)

        sz = min(2 * n, 2**(maxDepth + 1))   # pre-allocate storage for tree:
        L, R, F, T = np.zeros((sz,)), np.zeros((sz,)), np.zeros((sz,)), np.zeros((sz,))

        L, R, F, T, last = self.__dectree_train(X, Y, L, R, F, T, 0, 0, minParent, maxDepth, minScore, nFeatures)

        self.L = L[0:last]
        self.R = R[0:last]
        self.F = F[0:last]
        self.T = T[0:last]



    def predict(self, X):
        """
        This method makes predictions on the test data X.

        Parameters
        ----------
        X : M x N numpy array of M data points (N features each) at which to predict
        """
        Y_te = self.__dectree_test(X, self.L, self.R, self.F, self.T, 0).T.ravel()
        return arr([[self.classes[int(i)]] for i in np.ravel(Y_te)])
        # TODO: FIX + INEFFICIENT


## HELPERS #####################################################################


    def __dectree_train(self, X, Y, L, R, F, T, next, depth, minParent, maxDepth, minScore, nFeatures):
        """
        This is a recursive helper method that recusively trains the decision tree. Used in:
            train
        """
        n,d = np.asmatrix(X).shape      # get data shape
        num_classes = len(self.classes) # we'll assume Y is in 0..C-1 "index" form

        if n < minParent or depth >= maxDepth or np.all(Y == Y[0]):
            assert n != 0, ('TreeClassify.__dectree_train: tried to create size zero node')
            F[next] = -1
            #tmp = np.sum(to1ofK(Y, range(1, num_classes + 1)), axis=0)
            pr = np.sum(to1ofK(Y, self.classes), axis=0)  # compute class distribution
            T[next] = np.argmax(pr)     # and best prediction for this leaf
            next += 1
            return (L,R,F,T,next)

        best_val = -np.inf
        best_feat = -1
        try_feat = np.random.permutation(d)
        wts_left = np.arange(1.0*n)/n

        for i_feat in try_feat[0:nFeatures]:   # try "n" randomly selected features:
            dsorted = np.asarray(np.sort(X[:,i_feat].T)).ravel()
            pi = np.argsort(X[:,i_feat].T)
            tsorted = Y[pi].ravel()             # get targets, sorted by feature i_feat
            can_split = np.append(np.asarray(dsorted[:-1] != dsorted[1:]), 0)
            # TODO: include numerical tolerance in check

            if not np.any(can_split):           # no way to split on this feature?
                continue

            y_left = np.cumsum(to1ofK(tsorted, self.classes), axis=0).astype(float)
            y_right = y_left[-1,:] - y_left

            for i in range(n):
                y_left[i,:] /= i+1
                y_right[i,:] /= n-1-i if i < n-1 else 1.0

            h_root = np.dot(-y_left[-1,:], np.log(y_left[-1,:] + np.spacing(1)).T) 
            h_left = -np.sum(y_left * np.log(y_left + np.spacing(1)), axis=1)
            h_right = -np.sum(y_right * np.log(y_right + np.spacing(1)), axis=1)

            IG = h_root - (wts_left * h_left + (1.0-wts_left) * h_right)
            val = np.max((IG + np.spacing(1)) * can_split)
            index = np.argmax((IG + np.spacing(1)) * can_split)

            if val > best_val:
                best_val = val
                best_feat = i_feat
                best_thresh = (dsorted[index] + dsorted[index + 1]) / 2

        if best_feat == -1:    # if no viable features found, make a leaf:
            F[next] = -1
            tmp = np.sum(to1ofK(Y, self.classes), axis=0)
            nc = np.max(tmp)
            T[next] = np.argmax(tmp)
            next += 1
            return (L,R,F,T,next)

        F[next] = best_feat
        T[next] = best_thresh

        go_left = X[:,F[next]] < T[next]
        my_idx = next
        next += 1

        L[my_idx] = next
        L,R,F,T,next = self.__dectree_train(X[go_left,:], Y[go_left], L, R, F, T, 
            next, depth + 1, minParent, maxDepth, minScore, nFeatures)

        R[my_idx] = next
        L,R,F,T,next = self.__dectree_train(X[np.logical_not(go_left),:], Y[np.logical_not(go_left)], L, R, F, T, 
            next, depth + 1, minParent, maxDepth, minScore, nFeatures)

        return (L,R,F,T,next)


    def __dectree_test(self, X, L, R, F, T, pos):
        """
        This is a recursive helper method that finds class labels
        in the decision tree for prediction. Used in:
            predict
        """
        M,N = X.shape
        y_hat = np.zeros((M,1))

        if F[pos] == -1:
            y_hat[:] = T[pos]
        else:
            go_left = X[:,F[pos]] < T[pos]  # which data should follow left split?
            y_hat[go_left]  = self.__dectree_test(X[go_left,:],  L, R, F, T, L[pos])
            go_right = np.logical_not(go_left)  # other data go right:
            y_hat[go_right] = self.__dectree_test(X[go_right,:], L, R, F, T, R[pos])

        return y_hat


################################################################################
################################################################################
################################################################################

