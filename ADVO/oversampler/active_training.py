#Idea. Fit a multivariate Gaussian on the errors, make predictins for ADV-O as most likely fraud + draw from Gaussian.
#Then filter through classifier. Since no library allows to fit a multivariate normal dist, I am experimenting to do so.



import numpy as np
from numpy import *
import math
from scipy.stats import multivariate_normal





predictions = np.array([[2, 3, 4, 5, 6],  [2, 3, 4, 5, 6]]).transpose()
values =  np.array([[2.2, 3, 5, 5, 6], [2, 3, 4, 5, 6]]).transpose()
errors = values - predictions
x = errors

# mean vector
mu = np.array([2,3,8,10,3]).reshape(-1,1)
mu2 = [2,3,8,10,3]

# covariance matrix
sigma = matrix([[1, 0, 0, 0, 0.2],
           [0, 1, 0, 0, 0.3],
            [0.5, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0.3, 0, 0, 0, 1]
          ])

generator = multivariate_normal(mean= mu2, cov=sigma, allow_singular=True)
generator.rvs(10)

def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))

def test_gauss_pdf(errors, mu, sigma):
    x = errors
    mu  = mu
    cov = sigma

    print(pdf_multivariate_gauss(x, mu, cov))

test_gauss_pdf(errors=errors, mu=mu, sigma=sigma)