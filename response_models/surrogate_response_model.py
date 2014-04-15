from abcpy.response_models.gaussian_response_model import *
#from abcpy.helpers import log_pdf_full_mvn, log_pdf_diag_mvn, normcdf

import numpy as np
import pdb

class SurrogateResponseModel( GaussianResponseModel ):
  
  def load_params( self, params ):
    super(SurrogateResponseModel, self).__init__( params )
    
    self.surrogate = params["surrogate"]
   
  def new( self, params ):   
    m = SurrogateResponseModel( params )
    return m
  
  def is_empty( self ):
    return False
      
  def add( self, thetas, pseudo_statistics, observation_statistics ):
    if len( self.pseudo_statistics ) == 0:
      self.pseudo_statistics = pseudo_statistics.copy()
    else:
      self.pseudo_statistics = np.vstack( (self.pseudo_statistics, pseudo_statistics) )
      
    self.update()
      
  def make_estimatoes( self, theta ):
    # ignore thetas, observation_statistics
    
    # compute mean, stats_cov, mean_cov
    S,J = pseudo_statistics.shape
    assert S > 1, "must have at least 2 simulations"
    
    self.pstats_mean     = pseudo_statistics.mean(0)
    d = pseudo_statistics - self.pstats_mean 
    self.pstats_sumsq    = np.dot( d.T, d )
    self.pstats_cov      = self.pstats_sumsq / (S-1) + self.epsilon*np.eye(J)
    self.pstats_mean_cov = self.pstats_cov / S
    self.pstats_icov     = np.linalg.inv( self.pstats_cov )
    self.pstats_logdet   = -np.log( np.linalg.det( self.pstats_cov ) )  # neg cause of mvn parameterization
    
    # TODO: other options are to put priors
    
    # zero-out the off-diagonals;  make the statistics independent.
    if self.diagonalize:
      self.pstats_stddevs      = np.sqrt( np.diag( self.pstats_cov) )
      self.pstats_mean_stddevs = np.sqrt( np.diag( self.pstats_mean_cov) )
    
