"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np
#import pandas as pd
from math import log, sqrt

from .linear_classifier_f import LinearClassifier

def softmax(W,X): #softmax for all observations return (48000, 10)
    X = np.array(X)
    W = np.array(W)
    if(X is None): return (np.zeros([3073,10]))
    #print("softmax, X= ",X[1:2])
    #print("softmax, W.shape = ", W.shape)
    temp = np.exp(np.matmul(X,W))
    den = np.sum(temp, axis =1)    
    ans = np.divide(np.transpose(temp),den)    
    ans = np.transpose(ans)
    return(ans)
#def myOneHotEncoder(y):
#        s = pd.Series((y))
#        return(np.array(pd.get_dummies(s)))
#def myOneHotEncoder_small(y):
#        c = [0,1,2,3,4,5,6,7,8,9]
#        cat=pd.Series(list(y))
#        cat=cat.astype('category',categories=list(c))
    
#        obj = pd.get_dummies(cat)
#        obj = np.array(obj)
#        return(obj)

def myOneHotEncoder(y):   
    c = [0,1,2,3,4,5,6,7,8,9]
    y_enc = []
    for i in range(len(y)):
        for el in c:
            zero = np.zeros_like(c)
            if(el == y[i]):
                zero[el]=1
                break;
        y_enc.append(zero)
    return np.array(y_enc)  


    
def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    y_est = np.array(softmax(W, X))
    y_hot_enc = myOneHotEncoder(y)# 48 000 * 10
    #if(len(np.unique(y))<10):
    #    y_hot_enc = myOneHotEncoder_small(y)
      
    
    for c in range(W.shape[1]): # loop on classes 10 # 3073*10
        #y_est_i  - 1*10
        for i in range (X.shape[0]): # 48 000 * 3073 - features # loop on obs
            loss = loss + y_hot_enc[i][c]* log(y_est[i][c])   
            dW[:, c] =  dW[:, c] + (y_est[i][c] - y_hot_enc[i][c])*(X[i,:]) # 1* 3074
        
    
                
    loss=loss*(-1.0/X.shape[0])
    W_Frobenius = np.linalg.norm(W)
    loss = loss + reg*np.sum(np.power(W,2)) # FROBENIUS, LIKE EUCL FOR VECTORS

         
    dW= dW/X.shape[0]      
    dW = dW + 2*reg*W 
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    y_hot_enc = myOneHotEncoder(y)
    y_est = np.array(softmax(W, X))
    ln_y_est = np.log(y_est)
    
    tempMatrix= np.matmul(np.transpose(y_hot_enc), (ln_y_est))
    
    loss = np.trace(tempMatrix)
    loss = (-1.0/X.shape[0])* loss
    W_Frobenius = np.linalg.norm(W)
    loss = loss + reg*np.sum(np.power(W,2))
    
    dW = np.matmul(np.transpose(X),(y_est - y_hot_enc))
    dW= dW/X.shape[0]  
    dW = dW + 2*reg*W 
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]
    
    mystep = 1e-7
    step = 100
    lr = 1e-7
    rs=0.0
    rstep=5000
    best_key = -1
    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    ##for lrate in xrange(learning_rates[0], learning_rates[1] + mystep, mystep):
      ##   for rstrength in xrange(regularization_strengths[0], regularization_strengths[1] + step, step)
    
    while (lr <= 5e-7):

        while rs <=5e4:
            
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=10000)
            #print(loss)
            y_train_pred = softmax.predict(X_train)
            y_val_pred = softmax.predict(X_val)
            t_a = np.mean(y_train == y_train_pred)
            v_a = np.mean(y_val == y_val_pred)
            results[(lr,rs)]  = t_a, v_a
            
            best_val = max(best_val, v_a)
            if(best_val== v_a):
                best_key = (lr,rs)
                best_softmax = softmax
            all_classifiers.append(softmax)
            rs = rs + rstep
            
        lr= lr+1e-7
        rs = 0.0
    
        
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)
    print('best key: ', best_key)

    return best_softmax, results, all_classifiers
