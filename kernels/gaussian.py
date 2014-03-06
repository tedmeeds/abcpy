from abcpy.helpers import gaussian_logpdf

def log_gaussian_kernel( x, y, epsilon ):
  return gaussian_logpdf( x, y, epsilon  )