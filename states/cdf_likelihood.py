from abcpy.state import SyntheticLikelihoodState
from abcpy.helpers import logsumexp, log_pdf_full_mvn, gaussian_logpdf, inv_wishart_rnd, wishart_rnd, invgamma_rnd

import numpy as np
import scipy as sp
import pylab as pp

import pdb

# =========================================================================== #
#
#  CdfLikelihoodState: also calculates estimators
#
# =========================================================================== #
class CdfLikelihoodState( SyntheticLikelihoodState ):
  
  def __init__(self, theta, params ):
    super(CdfLikelihoodState, self).__init__(theta, params)
    self.stats    = []
    self.sim_outs = []
    
  def new( self, theta, params ):
    return CdfLikelihoodState( theta, params )             
      
  def mv_loglikelihood_under_model( self, stats, mu_stats, cov_stats ):
    return self.loglikelihood_under_model( stats, mu_stats, np.sqrt( np.diag(cov_stats)) )
    #return log_pdf_full_mvn( stats, mu_stats, cov_stats )
    
  def loglikelihood_under_model(self, stats, mu_stats, std_stats ): 
    logpdf = 0.0
    for s,mu,std in zip( stats, mu_stats, std_stats ):
      cdf = np.squeeze( normcdf( s, mu, std ) )
      if cdf ==0:
        logpdf += np.log(1e-12)
      else:
        logpdf += np.log( cdf )
    return logpdf