"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np
from math import log,log2,  sqrt

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape
        
        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################
        #print(" W1 = {0},\n W2 = {1},\n b1 = {2},\n b2 = {3},\n X = {4}\n \n".format(W1.shape, W2.shape, b1.shape, b2.shape, X.shape))
        #print(" W1 = {0},\n W2 = {1},\n b1 = {2},\n b2 = {3},\n X = {4}\n".format(W1, W2, b1, b2, X))
        #print("y  = ", y)
        
        
        bias1  = np.transpose(np.outer(b1, np.ones(N)))                  #5 obs N
        bias2  = np.transpose(np.outer(b2, np.ones(N)))        #Hidden layer H 10
        #print("bias1 {0}, bias2 {1}".format(bias1.shape, bias2.shape))
        
        #activ1 = np.matmul( X, W1 ) + bias1                  
        activ1 = np.matmul( X, W1 ) + bias1                 
        #print("activ1 {0} = {1}".format(activ1.shape,activ1))
        
        #the first layer ReLu
                          
        # in the vector case it is compared with zero,
        #so it means that it should be compared with zero elementwise in the matrix case
        hidden_layer = np.multiply(activ1, activ1>np.zeros_like(activ1)) # should be N* H = 5 times 10
        #print("hidden {0} = {1}".format(hidden_layer.shape,hidden_layer))
        #x*(x>0) = max{0, x}
        
        activ2 = np.matmul(hidden_layer, W2 ) + bias2
        #print("activ2 {0} = {1}".format(activ2.shape, activ2))                 
        
        #the second layer softmax
        temp = np.exp(activ2)
        scores = activ2
        
        den = np.sum(temp, axis =1)    # check the sum
        result_layer = np.transpose(np.divide(np.transpose(temp),den))# should give the matrix N times C 5 * 3
        #print("result l {0} = {1}".format(result_layer.shape, result_layer))   

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################
        
        
        ##########"""########################################
        ######################"""############################
        #################################"""#################
        #####################################################
        #print("some unknown problem is here. Loss.")
        #one hot encoder  
        c = np.sort(np.unique(y))
        y_enc = []
        for i in range(len(y)):
            for el in c:
                zero = np.zeros_like(c)
                if(el == y[i]):
                    zero[el]=1
                    break;
            y_enc.append(zero)
        y_hot_enc =  np.array(y_enc)  
        


        loss = np.trace(np.matmul(np.transpose(y_hot_enc), (np.log(result_layer)))) 
        loss = (-1.0/N)* loss
        loss = loss + reg*(   np.sum(np.power(W1,2)) + np.sum(np.power(W2,2))  )
        
        
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################

        ############# matrix form ##########
        #print("=====================  MATRIX   =================")
        #grad_loss =  (-1/N)* np.trace(np.dot((y_hot_enc), np.transpose(1.0/(result_layer)))) # c n n c
        
        """
        dresult_layer = 1.0/result_layer
       
        grad_loss =  (-1/N)* np.multiply(y_hot_enc, dresult_layer) # nc nc  should be nc
        grad_loss =  (-1/N)* np.multiply(y_hot_enc, dresult_layer) # nc nc  should be nc
        print(grad_loss)
        
        mygrad_loss = []
        for i in range(grad_loss.shape[0]):
            for j in range(grad_loss.shape[1]):
                if(grad_loss[i,j]!=0.0):
                    #grad_loss[i,j] = 0.0
                    mygrad_loss.append(grad_loss[i,j])
        #print("grad loss = {0}".format(grad_loss))             
        grad_loss = mygrad_loss
        #print("grad loss = {0}".format(grad_loss))
                
        
         #softmax 
        
      
         # n * c
        grad_loss  = np.outer(grad_loss, np.ones_like(result_layer[0]))
        #print("grad loss = {0}".format(grad_loss))
        print(" softmax = ", np.multiply((result_layer), 1 - result_layer))
        grad_y_pred  = np.multiply(grad_loss, np.multiply((result_layer), 1 - result_layer))
        """
        
        grad_y_pred = 1/N * (result_layer - y_hot_enc)
        
        #W2
        
        grad_w2 = np.dot(hidden_layer.T, grad_y_pred)# h * n* n *c
        grad_w2 = grad_w2 + reg*2*W2
        
        #grad_w2 = (1/N)* (np.matmul(np.transpose(hidden_layer),(result_layer - y_hot_enc)))
        #temp = (result_layer - y_hot_enc)/N
        #print("temp = ", temp)
        #grad_w2 = grad_w2 + reg*2*W2
        #print("grad_w2 {0} = \n {1}".format(grad_w2.shape, grad_w2 ))
        
        grad_b2 = (1/N)*(np.matmul(np.ones_like(result_layer).T,(result_layer - y_hot_enc))) + reg*2*b2
               
        prod = np.dot(grad_y_pred,W2.T) # grad of the node max{0, z0} i.e hidden layer node
        # n c c h = n h  
        grad_h = prod
       
        
        my_zeros = np.zeros_like(hidden_layer)       
        grad_relu = np.zeros_like(my_zeros)
        grad_relu[np.where(hidden_layer>my_zeros)] = prod[np.where(hidden_layer>my_zeros)] # 1 *h  
        
        
        #w1        
        grad_w1= np.matmul(X.T, grad_relu) # d * n* n * h
        grad_w1 = grad_w1 + reg*2*W1
        grad_b1 = np.matmul(np.ones_like(grad_relu).T, grad_relu) + reg*2*b1
        
        grads['W1'] = grad_w1
        grads['W2'] = grad_w2
        grads['b1'] = grad_b1[0]
        grads['b2'] = grad_b2[0]
        
        """
        print("grad loss =\n {0}".format(grad_loss))   
        print("grad_y_pred {0} =\n {1}".format(grad_y_pred.shape, grad_y_pred))
        print("grad_w2 {0} = \n {1}".format(grad_w2.shape, grad_w2 ))
        print("grad_h {0} = \n {1}".format(grad_h.shape, grad_h ))
        print("grad_relu {0} = {1}".format(grad_relu.shape, grad_relu ))
        print("grad_w1 {0} = \n {1}".format(grad_w1.shape, grad_w1 ))
        #"""
        
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################
            mini_batch_indices = np.random.choice(len(X),size = batch_size, replace = True)
            X_batch = X[mini_batch_indices]
            y_batch = y[mini_batch_indices]
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################
            """
            self.params['W1'] = self.params['W1'] - learning_rate*grads['W1']
            self.params['W2'] = self.params['W2'] - learning_rate*grads['W2']
            self.params['b1'] = self.params['b1'] - learning_rate*grads['b1']
            self.params['b2'] = self.params['b2'] - learning_rate*grads['b2']            
            """
            #adam 
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            mW1, mW2, mb1, mb2 = 0 ,0 ,0 ,0
            vW1, vW2, vb1, vb2 = 0, 0, 0, 0

            mW1 = beta1* mW1 + (1-beta1)* grads['W1']
            mW2 = beta1* mW2 + (1-beta1)* grads['W2']
            mb1 = beta1* mb1 + (1-beta1)* grads['b1']
            mb2 = beta1* mb2 + (1-beta1)* grads['b2']
            
            vW1 = beta2* vW1 + (1-beta2)* np.multiply(grads['W1'],grads['W1'])
            vW2 = beta2* vW2 + (1-beta2)* np.multiply(grads['W2'],grads['W2'])
            vb1 = beta2* vb1 + (1-beta2)* np.multiply(grads['b1'],grads['b1'])
            vb2 = beta2* vb2 + (1-beta2)* np.multiply(grads['b2'],grads['b2'])
            
            
            mW1_e = mW1/(1-beta1)
            mW2_e= mW2/(1-beta1)
            mb1_e = mb1/(1-beta1)
            mb2_e = mb2/(1-beta1)
            
            vW1_e = vW1/(1-beta2)
            vW2_e = vW2/(1-beta2)
            vb1_e = vb1/(1-beta2)
            vb2_e = vb2/(1-beta2)
            
            self.params['W1'] = self.params['W1'] - learning_rate*np.divide(mW1_e, np.sqrt(vW1_e) + eps)
            self.params['W2'] = self.params['W2'] - learning_rate*np.divide(mW2_e, np.sqrt(vW2_e) + eps)
            self.params['b1'] = self.params['b1'] - learning_rate*np.divide(mb1_e, np.sqrt(vb1_e) + eps)
            self.params['b2'] = self.params['b2'] - learning_rate*np.divide(mb2_e, np.sqrt(vb2_e) + eps) 
            
            
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        bias1  = np.transpose(np.outer(b1, np.ones(N)))             
        bias2  = np.transpose(np.outer(b2, np.ones(N)))   

      
        activ1 = np.matmul( X, W1 ) + bias1                 
        hidden_layer = np.multiply(activ1, activ1>np.zeros_like(activ1)) #relu x*(x>0) = max{0, x}       
        activ2 = np.matmul(hidden_layer, W2 ) + bias2
        temp = np.exp(activ2)      
        result_layer = np.transpose(np.divide(np.transpose(temp),np.sum(temp, axis =1))) #softmax
        y_pred = np.argmax(result_layer,axis = 1) 
        #print("bias1 {2} = {0}, bias2 {3}= {1}".format(bias1, bias2, bias1.shape, bias2.shape))
        #print("activ1 {0} = {1}".format(activ1.shape,activ1))
        #print("hidden {0} = {1}".format(hidden_layer.shape,hidden_layer))
        #print("activ2 {0} = {1}".format(activ2.shape, activ2)) 
        #print("result l {0} = {1}".format(result_layer.shape, result_layer))   
        
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 
    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################

    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    #learning_rates = [8e-4,8.5e-4, 9e-4, 9.5e-4]
    #regularization_strengths = [0.0]
    
    #learning_rates = [3e-4, 3.5e-3, 4e-4, 4.5e-4, 5e-4, 5.5e-4, 6e-4, 6.5e-4, 7e-4, 7.5e-4, 8e-4, 8.2e-4, 8.5e-4, 8.7e-4, 9e-4, 9.3e-4, 9.5e-4, 9.7e-4, 10e-4]
    #regularization_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #iterations = [1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000]
    #hs = [47,50,52,55,57,60,62,65,67,70,75,80,85,90,95,100,110,120,130,140,150,200,250,300,350,400,450,500]
    #rs= 0.0
    
    #learning_rates = [7e-4, 7.5e-4, 8e-4, 8.2e-4, 8.5e-4, 8.7e-4, 9e-4, 9.3e-4, 9.5e-4, 9.7e-4, 10e-4]
    #learning_rates = [8e-4, 8.2e-4, 8.5e-4, 8.7e-4, 9e-4, 9.3e-4, 9.5e-4, 9.7e-4]
    #regularization_strengths = [0.0]#, 0.1, 0.2, 0.3, 0.4, 0.5]
    #iterations = [1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000]
    #iterations = [2000]
    
    
    learning_rates = [0.0001]
    regularization_strengths = [0.1]
    iterations = [15000, 100]
    hs = [300]
    
    #learning_rates = [5e-5]
    #regularization_strengths = [0.1]
    #iterations = [20000, 100]
    #hs = [500]
    
    input_size = X_train.shape[1]
    num_classes = 10
    
    for rs in regularization_strengths:    
        for lr in learning_rates:        
            for it in iterations:
                for h in  hs:
                        print(lr,rs,it,h)
            
            
                        net = TwoLayerNet(input_size, h, num_classes)


                        # Train the network
                        stats = net.train(X_train, y_train, X_val, y_val, num_iters=it, batch_size=200, learning_rate=lr, learning_rate_decay=0.95,reg=rs, verbose=True)

                        #print(loss)
                        y_train_pred = net.predict(X_train)
                        y_val_pred = net.predict(X_val)
                        t_a = np.mean(y_train == y_train_pred)
                        v_a = np.mean(y_val == y_val_pred)
                        results[(lr,rs,it,h)]  = t_a, v_a
                        print('lr %e reg %e it %e h %e train accuracy: %f val accuracy: %f' % (
              lr, rs, it, h, t_a, v_a))
                        
                        best_val = max(best_val, v_a)
                        if(best_val== v_a):
                            best_key = (lr,rs,it,h)
                            best_net = net
                            #yield best_net
                        all_classifiers.append(net)
            
            
        
        # Print out results.
    #for (lr, reg,it, h) in sorted(results):
     #   train_accuracy, val_accuracy = results[(lr, reg,it, h)]
     #   print('lr %e reg %e it %e h %e train accuracy: %f val accuracy: %f' % (
     #         lr, reg, it, h, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)
    print('best key: ', best_key)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
