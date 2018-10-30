"""Linear Classifier Base Class."""
# pylint: disable=invalid-name
import numpy as np


class LinearClassifier(object):
    """Linear Classifier Base Class."""

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
       
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            mini_batch_indices = np.random.choice(len(X),size = batch_size, replace = False) 
            #3  = np.arrange(3)= [0,1,2]
            X_batch = X[mini_batch_indices]
            y_batch = y[mini_batch_indices]
            #print(mini_batch_indices)
            #print("X, y shapes in batches= ", X_batch.shape, y_batch.shape)
            
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            
            loss, grad = self.loss(X_batch, y_batch, reg)
            #print("loss = {0}, grad = {1}, X_batch = {2}, y_batch = {3}".format(loss, grad, X_batch, y_batch))
            #print("```````````````````+++++++++++++++++++++++++++``````````````````````")
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
             #adam 
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            mW = 0 
            vW = 0

            mW = beta1* mW + (1-beta1)* grad            
            vW = beta2* vW + (1-beta2)* np.multiply(grad, grad)

            mW_e = mW/(1-beta1)           
            vW_e = vW/(1-beta2)

            self.W = self.W - learning_rate*np.divide(mW_e, np.sqrt(vW_e) + eps)
            
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                #print("my weights = >" , self.W[0:1,:])
                #print(grad)
                

        return loss_history
    
    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each row is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        #print("self.W = ", self.W)        
        temp = np.exp(np.matmul(X,self.W))
        den = np.sum(temp, axis =1)    
        ans = np.divide(np.transpose(temp),den)    
        y_est = np.transpose(ans)
        #print(y_est)
        y_pred = np.argmax(y_est,axis = 1) 
        #print("y_pred=", y_pred)
        
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        raise NotImplementedError
