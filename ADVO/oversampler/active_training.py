#Idea. Fit a multivariate Gaussian on the errors,
# make predictins for ADV-O as most likely fraud + draw from Gaussian.
#Finally, filter through classifier.

import numpy as np
from numpy import *
from scipy.stats import multivariate_normal

'''Placeholder for the module importing the predictions and true frauds from the ADV-O training set'''
def errors_placeholder():
    predictions = np.array([[2, 3], [3.3, 2.1], [0, 0.2]])
    values = np.array([[1, 3], [2, 3], [0.2, 1]])
    return values - predictions

errors = errors_placeholder()


mean = np.mean(errors, axis=0)
cov = np.cov(errors, rowvar=0)
generator = multivariate_normal(mean= mean, cov=cov, allow_singular=True)
print(generator.rvs(10))

