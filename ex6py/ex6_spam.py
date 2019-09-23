## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
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
import processEmail
import emailFeatures


## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.
print('\nPreprocessing sample email (emailSample1.txt)\n');

# Extract Features
with open('emailSample1.txt') as fid:
    file_contents = fid.read()

word_indices  = processEmail.proEmail(file_contents)

#Print Stats
print('-------------')
print('Word Indices:')
print('-------------')
print(word_indices)

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('\nExtracting features from sample email (emailSample1.txt)\n');

# Extract Features
with open('emailSample1.txt') as fid:
    file_contents = fid.read()
    word_indices  = processEmail.proEmail(file_contents)
    features      = emailFeatures.features(word_indices)

# Print Stats
print('\nLength of feature vector: %d' % len(features))
print('Number of non-zero entries: %d' % sum(features > 0))


## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data = loadmat('spamTrain.mat')
X, y= data['X'].astype(float), data['y'][:, 0]

print('Training Linear SVM (Spam Classification)')
print('This may take 1 to 2 minutes ...\n')

C = 0.1
model = utils.svmTrain(X, y, C, utils.linearKernel)

# Compute the training accuracy
p = utils.svmPredict(model, X)
print('Training Accuracy: %.2f' % (np.mean(p == y) * 100))

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
data = loadmat('spamTest.mat')
Xtest, ytest= data['Xtest'].astype(float), data['ytest'][:, 0]

print('\nEvaluating the trained Linear SVM on a test set ...\n')

# Compute the training accuracy
p = utils.svmPredict(model, Xtest)
print('Training Accuracy: %.2f' % (np.mean(p == ytest) * 100))

# ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtin the vocabulary list
idx = np.argsort(model['w'])
top_idx = idx[-15:][::-1]
vocabList = utils.getVocabList()

print('Top predictors of spam:')
print('%-15s %-15s' % ('word', 'weight'))
print('----' + ' '*12 + '------')
for word, w in zip(np.array(vocabList)[top_idx], model['w'][top_idx]):
    print('%-15s %0.2f' % (word, w))

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned SVM classifier to determine whether the email is Spam or
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = os.path.join('emailSample1.txt')

# Read and predict
with open(filename) as fid:
    file_contents = fid.read()
word_indices = processEmail.proEmail(file_contents, verbose=False)
x = emailFeatures.features(word_indices)
p = utils.svmPredict(model, x)

print('\nProcessed %s\nSpam Classification: %s' % (filename, 'spam' if p else 'not spam'))
