import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  train_size = X.shape[0]
  class_size = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(train_size):
    soft_sum = np.sum(np.exp(X[i,:].dot(W)))
    loss += -X[i,:].dot(W[:,y[i]]) + np.log(soft_sum)
    
    for j in range(class_size):
        dW[:, j] += np.exp(X[i].dot(W[:,j])) / soft_sum * X[i]
        if (j == y[i]):
            dW[:, j] -= X[i]
                           
    # tmp = np.exp(X[i,:].dot(W))/soft_sum - (range(class_size) == y[i]).astype(int)
    # tmp = tmp.reshape(1,-1)
    # dW += X[i][:,np.newaxis].dot(tmp)
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= train_size
  dW /= train_size
  loss += reg * np.sum(W*W)
  dW += 2 * reg * W
                          
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  train_size = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.exp(X.dot(W))
  scores_sum = np.sum(scores, axis=1)
  loss += -np.sum(np.log(scores[range(train_size), y] / scores_sum)) / train_size + reg * np.sum(W*W)
  
  tmp = scores / scores_sum[:, np.newaxis]
  tmp[range(train_size), y] -= 1
  dW += X.T.dot(tmp) / train_size + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

