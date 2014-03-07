import numpy as np
import scipy as sp
import pylab as pp

class BaseAllStates( object ):
  def __init__(self, keep_invalid = False):
    self.thetas          = []
    self.states          = []
    self.accepted        = []
    self.nbr_acceptances = 0
    self.sim_calls       = []
    self.nbr_sim_calls   = 0
  
    self.statistics      = None
    self.observations    = None
    
    self.invalid_thetas   = []
    self.keep_invalid     = keep_invalid
  
  def rejection_rate( self ):
    return 1.0 - self.acceptance_rate()
  
  def acceptance_rate( self ):
    return float( self.nbr_acceptances ) / float( len(self.accepted) )
  
  def acceptance_rate_per_simulation(self):
    return float( self.nbr_acceptances ) / float( self.nbr_sim_calls )
    
  def get_thetas( self, burnin = 0, accepted_only = False ):
    return np.array( self.thetas )[burnin:,:]
      
  def get_statistics( self, burnin = 1):
    if self.statistics is None:
      self.statistics = []
      for state in self.states:
        self.statistics.append( state.statistcs)
      self.statistics = np.array(self.statistics)
    return self.statistics
  
  def get_sim_calls(self):
    return np.array( self.sim_calls )
    
  def get_acceptances(self):
    return np.array( self.accepted )
     
  def get_states( self, burnin = 1 ):
    return self.states[burnin:]
       
  def add( self, state, nbr_sim_calls, accepted = True, other_state = None ):
    self.nbr_sim_calls += nbr_sim_calls
    self.accepted.append( accepted )
    self.sim_calls.append( nbr_sim_calls )
    self.thetas.append( state.theta )
    if accepted:
      self.nbr_acceptances += 1
      
  def add_invalid( self, state ):
    if self.keep_invalid:
      self.invalid_thetas.append( state.thetas )