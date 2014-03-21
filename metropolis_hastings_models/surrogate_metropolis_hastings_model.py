from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel
import numpy as np
import pylab as pp
import pdb

class DummyState( object ):
  def __init__( self, theta, statistics, stats = [] ):
    self.theta = theta
    self.statistics = statistics
    self.stats = stats 
    
class SurrogateMetropolisHastingsModel( AdaptiveMetropolisHastingsModel ):
  """ Provide an wrapper around surrogate model calls for Metropolis-Hastings models """  
  
  def load_params( self, params ):
    super(SurrogateMetropolisHastingsModel, self).load_params(params)
    self.surrogate = params["surrogate"]
  
  def reset_nbr_sim_calls_this_iter(self):
    self.surrogate.nbr_sim_calls_this_iter = 0
  
  def get_nbr_sim_calls_this_iter(self):
    return self.surrogate.nbr_sim_calls_this_iter
  
  def log_posterior(self):
    return self.surrogate.loglikelihood(self.current.theta) + self.current.logprior()
      
  def loglik_differences_rand( self, M ):   
    """ sample loglikelihood differences at proposed and current theta, use M samples """
    return self.surrogate.loglik_differences_rand( self.proposed.theta, self.current.theta, M )
        
  def acquire_points( self ):
    """ allow the surrogate to pick self.deltaS new simulation points """
    self.surrogate.acquire_points( self.proposed.theta, self.current.theta, self.deltaS )
    
  def stay_in_current_state( self ):
    if self.recorder is not None:
      #self.current.statistics = self.surrogate.offline_simulation( self.current.theta )
      #self.current.stats = self.current.statistics
      ds = DummyState(self.current.theta, np.squeeze(self.surrogate.offline_simulation( self.current.theta )))
      self.recorder.record_state( ds, self.get_nbr_sim_calls_this_iter(), accepted = False )   
      
  def move_to_proposed_state( self ):
    #pdb.set_trace()
    self.set_current_state( self.proposed )
    if self.recorder is not None:
      #self.current.statistics = self.surrogate.offline_simulation( self.current.theta )
      #self.current.stats = self.current.statistics
      ds = DummyState(self.current.theta, np.squeeze(self.surrogate.offline_simulation( self.current.theta )))
      self.recorder.record_state( ds, self.get_nbr_sim_calls_this_iter(), accepted = True ) 
  