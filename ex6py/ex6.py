## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.py
#     dataset3Params.py
#     processEmail.py
#     emailFeatures.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Import regular expressions to process emails
import re

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils
import gaussianKernel
import dataset3Params

# define the submission/grader object for this exercise
#grader = utils.Grader()

# tells matplotlib to embed plots within the notebook
#matplotlib inline

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data1:
# You will have X, y in your environment
data = loadmat('ex6data1.mat')
X, y = data['X'], data['y'][:, 0]

# Plot training data
utils.plotData(X, y)
pyplot.show()

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

# Load from ex6data1:
# You will have X, y in your environment
data = loadmat('ex6data1.mat')
X, y = data['X'], data['y'][:, 0]

print('\nTraining Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = utils.svmTrain(X, y, C, utils.linearKernel, 1e-3, 20)
utils.visualizeBoundaryLinear(X, y, model)
pyplot.show()

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('\nEvaluating the Gaussian Kernel ...\n')


x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel.gaussKer(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
      '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data2:
# You will have X, y in your environment
data = loadmat('ex6data2.mat')
X, y = data['X'], data['y'][:, 0]

# Plot training data
utils.plotData(X, y)
pyplot.show()

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the
#  SVM classifier.
#
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

# SVM Parameters
C = 1
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
model= utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils.visualizeBoundary(X, y, model)
pyplot.show()

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data3:
# You will have X, y in your environment
data = loadmat('ex6data3.mat')
X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]

# Plot training data
utils.plotData(X, y)
pyplot.show()

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
#

# Try different SVM Parameters here
C, sigma = dataset3Params.data3Parm(X, y, Xval, yval)

# Train the SVM
model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils.visualizeBoundary(X, y, model)
pyplot.show()
print(C, sigma)
