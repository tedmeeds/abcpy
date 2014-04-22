from abcpy.response_models.gaussian_response_model import *
#from abcpy.helpers import log_pdf_full_mvn, log_pdf_diag_mvn, normcdf

import numpy as np
import pdb

class SurrogateResponseModel( GaussianResponseModel ):
  
  def load_params( self, params ):
    super(SurrogateResponseModel, self).load_params( params )
    
    self.surrogate = params["surrogate"]
   
  def new( self, params ):   
    m = SurrogateResponseModel( params )
    return m
  
  def is_empty( self ):
    return self.surrogate.is_empty()
      
  def add( self, thetas, pseudo_statistics, observation_statistics ):
    if len( self.pseudo_statistics ) == 0:
      #self.pseudo_statistics = pseudo_statistics.copy()
      self.surrogate.add( thetas, pseudo_statistics )
    else:
      #self.pseudo_statistics = np.vstack( (self.pseudo_statistics, pseudo_statistics) )
      self.surrogate.add( thetas, pseudo_statistics )
      
    self.surrogate.update()
      
  def make_estimators( self, theta = None ):
    # in future, may want to memoize conditional distributions
    self.surrogate.make_estimators( theta )
    
  def loglikelihood( self, theta, observations ):
    self.make_estimators( theta )
    
    if self.likelihood_type == "logpdf":
      return self.surrogate.logpdf( theta, observations )
    elif self.likelihood_type == "logcdf":
      return self.surrogate.logcdf( theta, observations )
    elif self.likelihood_type == "logcdfcomplement":
      return self.surrogate.logcdfcomplement( theta, observations )
    else:
      raise NotImplementedError
    
  def loglikelihood_rand( self, theta, observations, N = 1 ):
    self.make_estimators( theta )
    
    if self.likelihood_type == "logpdf":
      random_logliks = self.surrogate.logpdf_rand( theta, observations, N )
    elif self.likelihood_type == "logcdf":
      random_logliks = self.surrogate.logcdf_rand( theta, observations, N )
    elif self.likelihood_type == "logcdfcomplement":
      random_logliks = self.surrogate.logcdfcomplement_rand( theta, observations, N )
    else:
      raise NotImplementedError
      
    return random_logliks
    
  def update( self ):
    self.surrogate.update()