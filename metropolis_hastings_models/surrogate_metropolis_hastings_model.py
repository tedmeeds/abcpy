from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel
import numpy as np
import pylab as pp
import pdb

class SurrogateMetropolisHastingsModel( AdaptiveMetropolisHastingsModel ):
  """ Provide an wrapper around surrogate model calls for Metropolis-Hastings models """  
  
  def load_params( self, params ):
    super(SurrogateMetropolisHastingsModel, self).load_params(params)
    self.surrogate = params["surrogate"]
  
  def reset_nbr_sim_calls_this_iter(self):
    self.surrogate.nbr_sim_calls_this_iter = 0
  
  def get_nbr_sim_calls_this_iter(self):
    return self.surrogate.nbr_sim_calls_this_iter
    
  def loglik_differences_rand( self, M ):   
    """ sample loglikelihood differences at proposed and current theta, use M samples """
    return self.surrogate.loglik_differences_rand( self.proposed.theta, self.current.theta, M )
        
  def acquire_points( self ):
    """ allow the surrogate to pick self.deltaS new simulation points """
    self.surrogate.acquire_points( self.proposed.theta, self.current.theta, self.deltaS )