from abcpy.abc_state import ABC_State
from abcpy.helpers import logsumexp
import numpy as np
import pdb

class KernelState(ABC_State):
  def __init__( self, params = None, response_groups = None ):
    super(KernelState, self).__init__(params,response_groups)
    
    self.loglikelihood_is_computed   = False
    self.discrepancy_values            = None
    
    if params.has_key("kernel"):
      self.kernel = params["kernel"]
    
  def new( self, theta, params = None, response_groups = None ):
    if theta is None:
      theta = self.theta
    if params is None:
      params = self.params
    if response_groups is None:
      response_groups = self.response_groups
    s = KernelState( params, response_groups )
    s.set_theta(theta)
    return s
  
  def update_post_mh(self):
    if len(self.simulation_statistics)==0:
      return
      
    ngroups = len(self.observation_groups)
    for group_id, og, rg in zip( range(ngroups), self.observation_groups, self.response_groups ):
      rg.update_post_mh( og, self.simulation_statistics[:,og.ids], self.params )
      og.update_post_mh( rg, self.simulation_statistics[:,og.ids], self.params )
     
  def loglikelihood(self):
    if self.loglikelihood_is_computed:
      return self.loglikelihood_value
      
    #pdb.set_trace()
    self.run_simulator_and_compute_statistics()
    self.compute_loglikelihood()
    
    return self.loglikelihood_value
  
  def compute_loglikelihood(self):
    statistics   = self.simulation_statistics
    observations = self.observation_statistics   
    
    S,J1 = statistics.shape
    N,J = observations.shape
    
    assert J == J1, "observation stats and pseudo stats should be the same"
    
    self.loglikelihood_value = 0.0
    
    # each row is the average discrepancy between observations and pseuo stats
    self.discrepancy_values = np.zeros( (N,J) )
    
    ngroups = len(self.observation_groups)
    logkernel = np.zeros( (S,ngroups) ) - np.log(S)
    for group_id, sg, rg in zip( range(ngroups), self.observation_groups, self.response_groups ):
      
      # over all pseudo statistics
      for s in range(S):
        # ystar = observation statistics for this stats group
        #pdb.set_trace()  
        logkernel[s,group_id] = rg.loglikelihood( sg.ystar, statistics[s,sg.ids] )
    
    #print "logkernel: ", logkernel 
        
    # sum log probs across ngroups statistics  
    loglike_by_s = logkernel.sum(1) - np.log(S)
    #pdb.set_trace()

    #   
    # loglikelihood for this observation is log sum_s=1^S exp( sum_j log(p(y_star_j | y_s)))
    loglike_n = logsumexp( loglike_by_s )
    
    #print loglike_n, loglike_by_s, logkernel
    #pdb.set_trace()
    # logsumexp will return Nan when the max value is -inf
    if np.isnan(loglike_n):
      loglike_n = -np.inf
    
    self.loglikelihood_value = loglike_n
      
    self.loglikelihood_is_computed = True
    
  # def loglikelihood( self ):
  #   return self.response_model.loglikelihood()
  #   
  # def loglikelihood_rand( self, M=1 ):
  #   return self.response_model.loglikelihood_rand(M)