#Idea. Fit a multivariate Gaussian on the errors, make predictins for ADV-O as most likely fraud + draw from Gaussian.
#Then filter through classifier. Since no library allows to fit a multivariate normal dist, I am experimenting to do so.



import numpy as np
from numpy import *
import math
from scipy.stats import multivariate_normal

predictions = np.array([[2, 3],  [3.3, 2.1], [0, 0.2]])
values =  np.array([[1, 3], [2, 3], [0.2, 1]])
errors = values - predictions

mean = np.mean(errors, axis=0)
cov = np.cov(errors, rowvar=0)


generator = multivariate_normal(mean= mean, cov=cov, allow_singular=True)
print(generator.rvs(10))

