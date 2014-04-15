import numpy as np
import pdb

class ABC_State(object):
  def __init__( self, theta, params, response_groups = None ):
    self.theta         = theta
    self.params        = params
    self.nbr_sim_calls = 0
    self.nbr_sim_calls_this_iter = 0
    
    if params.has_key("S"):
      self.S    = params["S"]
    else:
      self.S             = 1
    self.D = None
    self.observation_statistics = params["observation_statistics"]
    self.simulation_function    = params["simulation_function"]
    self.statistics_function    = params["statistics_function"]
    self.observation_groups     = params["observation_groups"]
    if response_groups is None:
      self.response_groups        = params["response_groups"]
    else:
      self.response_groups        = response_groups
  
    try:
      self.N, self.J = self.observation_statistics.shape
    except:
      self.N = len(self.observation_statistics)
      self.J = 1
      self.observation_statistics = self.observation_statistics.reshape( (self.N,self.J))
    
  def add_sim_call( self, N=1 ):
    self.nbr_sim_calls           += N
    self.nbr_sim_calls_this_iter += N
  
  def reset_nbr_sim_calls_this_iter(self):
    self.nbr_sim_calls_this_iter = 0
    
  def loglikelihood( self ):
    raise NotImplementedError
    
  def loglikelihood_rand( self, M=1 ):
    raise NotImplementedError
    
  def run_simulator_and_compute_statistics(self, reset = True, S = None ):
    if S is None:
      S = self.S
      
    # keep track of old results (eg for acquisition functions -- adding new design pts)
    if reset is False:
      old_sim_outputs  = self.simulation_outputs
      old_sim_stats = self.simulation_statistics
      
    # resetting outputs and statistics (save these before if necessary)
    self.simulation_outputs     = []
    self.simulation_statistics  = []
    
    # sometimes there may be more that one simulation run, this must be set in params
    for s in range(S):
      # simulation -> outputs -> statistics
      self.simulation_outputs.append( np.squeeze( self.simulation_function( self.theta ) ) )
      
      if self.D is None:
        #pdb.set_trace()
        if self.simulation_outputs[-1].__class__ == np.ndarray:
          self.D = len(self.simulation_outputs[-1])
        else:
          self.D = 1
          
      # keep track of simulation calls
      self.add_sim_call()
      
      # process for statistics
      self.simulation_statistics.append( self.statistics_function( self.simulation_outputs[-1] ) )
      
    # make into arrays
    self.simulation_outputs    = np.array(self.simulation_outputs).reshape( (S,self.D))
    self.simulation_statistics = np.array(self.simulation_statistics).reshape( ( S, self.J ) )
    
    if reset is False:
      self.simulation_outputs    = np.vstack( (old_sim_outputs,self.simulation_outputs))
      self.simulation_statistics = np.vstack( (old_sim_stats,self.simulation_statistics))