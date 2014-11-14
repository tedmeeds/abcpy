from abcpy.abc_state import ABC_State
import numpy as np

class DiscrepancyState(ABC_State):
  def __init__( self, params = None ):
    super(DiscrepancyState, self).__init__(params)
    
    self.discrepancies_are_computed   = False
    self.discrepancy_values = None
    
  def new( self, theta, params = None ):
    if theta is None:
      theta = self.theta
    if params is None:
      params = self.params
    s = DiscrepancyState( params )
    s.set_theta(theta)
    return s
   
  def discrepancy(self):
    if self.discrepancies_are_computed:
      return self.discrepancy_values
      
    self.run_simulator_and_compute_statistics()
    self.compute_discrepancies()
    
    return self.discrepancy_values
  
  def compute_discrepancies(self):
    statistics   = self.simulation_statistics
    observations = self.observation_statistics   
    
    S,J1 = statistics.shape
    N,J = observations.shape
    
    assert J == J1, "observation stats and pseudo stats should be the same"
    
    # each row is the average discrepancy between observations and pseuo stats
    self.discrepancy_values = np.zeros( (N,J) )
    
    # over all observation statistics
    for n in range(N):
      d = np.zeros( (1,J) )
      
      # over all pseudo statistics
      for s in range(S):
        d += statistics[s,:] - observations[n,:]
        
      # mean difference  
      d /= S
      self.discrepancy_values[n,:] = d
      
    self.discrepancies_are_computed = True
    
  def loglikelihood( self ):
    return self.response_model.loglikelihood()
    
  def loglikelihood_rand( self, M=1 ):
    return self.response_model.loglikelihood_rand(M)