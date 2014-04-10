import numpy as np
import pdb

class ABC_State(object):
  def __init__( self, theta, params ):
    self.theta         = theta
    self.params        = params
    self.nbr_sim_calls = 0
    self.nbr_sim_calls_this_iter = 0
    self.S             = 1
    
    self.observation_statistics = params["observation_statistics"]
    self.simulation_function = params["simulation_function"]
    self.statistics_function = params["statistics_function"]
  
    try:
      self.N, self.J = self.observation_statistics.shape
    except:
      self.N = len(self.observation_statistics)
      self.J = 1
      self.observation_statistics = self.observation_statistics.reshape( (self.N,self.J))
    
  def add_sim_call( self, N=1 ):
    self.nbr_sim_calls           += N
    self.nbr_sim_calls_this_iter += N
    
  def loglikelihood( self ):
    raise NotImplementedError
    
  def loglikelihood_rand( self, M=1 ):
    raise NotImplementedError
    
  def run_simulator_and_compute_statistics(self):
    # resetting outputs and statistics (save these before if necessary)
    self.simulation_outputs     = []
    self.simulation_statistics  = []
    
    # sometimes there may be more that one simulation run, this must be set in params
    for s in range(self.S):
      # simulation -> outputs -> statistics
      self.simulation_outputs.append( np.squeeze( self.simulation_function( self.theta ) ) )
      
      # keep track of simulation calls
      self.add_sim_call()
      
      # process for statistics
      self.simulation_statistics.append( self.statistics_function( self.simulation_outputs[-1] ) )
      
    # make into arrays
    self.simulation_outputs    = np.array(self.simulation_outputs)
    self.simulation_statistics = np.array(self.simulation_statistics).reshape( ( self.S, self.J ) )