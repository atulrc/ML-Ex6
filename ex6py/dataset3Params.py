#from submission import SubmissionBase
import numpy as np

import utils
import gaussianKernel

def data3Parm(X, y, Xval, yval):
    """
    Returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel.

    Parameters
    ----------
    X : array_like
        (m x n) matrix of training data where m is number of training examples, and
        n is the number of features.

    y : array_like
        (m, ) vector of labels for ther training data.

    Xval : array_like
        (mv x n) matrix of validation data where mv is the number of validation examples
        and n is the number of features

    yval : array_like
        (mv, ) vector of labels for the validation data.

    Returns
    -------
    C, sigma : float, float
        The best performing values for the regularization parameter C and
        RBF parameter sigma.

    Instructions
    ------------
    Fill in this function to return the optimal C and sigma learning
    parameters found using the cross validation set.
    You can use `svmPredict` to predict the labels on the cross
    validation set. For example,

        predictions = svmPredict(model, Xval)

    will return the predictions on the cross validation set.

    Note
    ----
    You can compute the prediction error using

        np.mean(predictions != yval)
    """
    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3



    # ====================== YOUR CODE HERE ======================

    C_array = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigma_array = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

    err_array = np.zeros([C_array.size, sigma_array.size])

    for i in np.arange(C_array.size):
        for j in np.arange(sigma_array.size):
            model= utils.svmTrain(X, y, C_array[i], gaussianKernel, args=(sigma_array[j],))
            predictions = utils.svmPredict(model, Xval)
            pred_error = np.mean(predictions != yval)
            print('pred_error', pred_error)
            err_array[i, j] = pred_error

    ind = np.unravel_index(np.argmin(err_array, axis = None), err_array.shape)
    C = C_array[ind[0]]
    sigma = sigma_array[ind[1]]

    # ============================================================
    return C, sigma
