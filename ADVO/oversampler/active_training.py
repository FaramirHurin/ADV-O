#Idea. Fit a multivariate Gaussian on the errors,
# make predictins for ADV-O as most likely fraud + draw from Gaussian.
#Finally, filter through classifier. This means, train a classifier on data before oversampling, remove most likely genuine
#among the supposed genuine frauds. To compare with standard ADVO

import numpy as np
from numpy import *
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier

'''Placeholder for the module importing the predictions and true frauds from the ADV-O training set'''
def errors_placeholder():
    predictions = np.array([[2, 3], [3.3, 2.1], [0, 0.2]])
    values = np.array([[1, 3], [2, 3], [0.2, 1]])
    return values - predictions

def classifier_placeHolder():
    data = None
    trained_classifier = RandomForestClassifier()
    trained_classifier.fit()
    return trained_classifier

errors = errors_placeholder()
trained_classifier = classifier_placeHolder()




mean = np.mean(errors, axis=0)
cov = np.cov(errors, rowvar=0)
generator = multivariate_normal(mean= mean, cov=cov, allow_singular=True)
print(generator.rvs(10))



# Fit random noise on the errors of the predictions
# Generate data as optimal prediction + draw from noise
# Use data selection method (based on feature selection) to select the best subset for classifier performance
