from abcpy.states.kernel_based_state import KernelState
import numpy as np
import pdb

class ResponseModelState(KernelState):
  def __init__( self, theta, params, response_model = None ):
    super(ResponseModelState, self).__init__(theta, params )
    if response_model is None:
      self.response_model = self.params["response_model"]
    else:
      self.response_model = response_model
    
  def new( self, theta, params = None ):
    if theta is None:
      theta = self.theta
    if params is None:
      params = self.params
      
    # response_model will decide if new means copy or just set to same response model (ie for surrogates)
    response_model = self.response_model.new( params )
    return ResponseModelState( theta, params, response_model )
   
  def acquire( self, N = 1 ):
    # run for N more times, do not reset stats already computed
    self.run_simulator_and_compute_statistics( reset = False, S = N )
    self.loglikelihood_is_computed = False
    
  # def loglikelihood(self):
  #   if self.loglikelihood_is_computed:
  #     return self.loglikelihood_value
  #     
  #   self.run_simulator_and_compute_statistics()
  #   self.compute_loglikelihood()
  #   
  #   return self.loglikelihood_value
  
  def compute_loglikelihood(self):
    pseudo_statistics      = self.simulation_statistics
    observation_statistics = self.observation_statistics   
    
    self.response_model.update( self.theta, pseudo_statistics, observation_statistics )
    
    S,J1 = pseudo_statistics.shape
    N,J  = observation_statistics.shape
    
    assert J == J1, "observation stats and pseudo stats should be the same"
    
    # over all observation statistics
    pdb.set_trace
    loglike_n = self.response_model.loglikelihood( observation_statistics )
        
    self.loglikelihood_value = loglike_n.sum()
      
    self.loglikelihood_is_computed = True
    
  def loglikelihood_rand( self, M=1 ):
    # call likelihood to force running simulator
    loglik = self.loglikelihood()
      
    return self.response_model.loglikelihood_rand( self.observation_statistics, M )
      
  # def loglikelihood( self ):
  #   return self.response_model.loglikelihood()
  #   
  # def loglikelihood_rand( self, M=1 ):
  #   return self.response_model.loglikelihood_rand(M)