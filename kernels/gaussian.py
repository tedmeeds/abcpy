from abcpy.helpers import gaussian_logpdf
import numpy as np

def log_gaussian_kernel( x, y, epsilon ):
  # will require a multivariate solution for full cov epsilon
  return np.sum(gaussian_logpdf( x, y, epsilon  ))