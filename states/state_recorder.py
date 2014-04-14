import numpy as np
import scipy as sp
import pylab as pp
import pdb

class BaseStateRecorder( object ):
  def __init__(self, keep_invalid = False, record_stats = False):
    self.thetas          = []
    self.states          = []
    self.accepted        = []
    self.nbr_acceptances = 0
    self.sim_calls       = []
    self.nbr_sim_calls   = 0
  
    self.record_stats = record_stats
    self.statistics      = []
    self.mu_statistics   = []
    self.observations    = []
    
    self.invalid_thetas   = []
    self.keep_invalid     = keep_invalid
  
  def rejection_rate( self ):
    return 1.0 - self.acceptance_rate()
  
  def acceptance_rate( self ):
    return float( self.nbr_acceptances ) / float( len(self.accepted) )
  
  def acceptance_rate_per_simulation(self):
    return float( self.nbr_acceptances ) / float( self.nbr_sim_calls )
    
  def get_thetas( self, burnin = 0, accepted_only = False ):
    return np.squeeze(np.array( self.thetas ))[burnin:]
      
  def get_stats( self, burnin = 0):
    
    if len(self.statistics)>0:
      #self.statistics = np.squeeze(np.array(self.statistics))
      return np.squeeze(np.array(self.statistics))[burnin:]
    else:
      return np.squeeze(np.array(self.statistics))
      
  def get_statistics( self, burnin = 0):
    #self.mu_statistics = np.squeeze(np.array(self.mu_statistics))
    if len(self.mu_statistics)>0:
      return np.squeeze(np.array(self.mu_statistics))[burnin:]
    else:
      return np.array(self.mu_statistics)
  
  def get_sim_calls(self, burnin = 0):
    return np.array( self.sim_calls )[burnin:]
    
  def get_acceptances(self):
    return np.array( self.accepted )
     
  def get_states( self, burnin = 0 ):
    return self.states[burnin:]
  
  def get_invalid( self):
    return np.array( self.invalid_thetas )
         
  def record_state( self, state, nbr_sim_calls, accepted = True, other_state = None ):
    #if nbr_sim_calls > 4:
    #  pdb.set_trace()
    self.nbr_sim_calls += nbr_sim_calls
    self.accepted.append( accepted )
    self.sim_calls.append( nbr_sim_calls )
    self.thetas.append( state.theta )
    if self.record_stats:
      
      self.mu_statistics.append(state.simulation_statistics)
      # if len( state.stats) > 0:
#         if len(self.statistics) == 0 :
#           #self.statistics.append( state.stats )
#           self.statistics = np.squeeze(np.array( state.stats ))
#         else:
#           self.statistics = np.vstack( (self.statistics, np.squeeze(np.array( state.stats )) ))
    if accepted:
      self.nbr_acceptances += 1
      
  def record_invalid( self, state ):
    if self.keep_invalid:
      self.invalid_thetas.append( state.thetas )
      
  def save_results( self, file_root ):
    thetas         = np.squeeze(self.get_thetas() )
    stats          = np.squeeze(self.get_statistics())
    sim_stats          = np.squeeze(self.get_stats())
    acceptances    = np.squeeze(self.get_acceptances())
    sims           = np.squeeze(self.get_sim_calls())
    invalid_thetas = np.squeeze(self.get_invalid())
    
    np.savetxt( file_root + "_thetas.txt", thetas, fmt='%0.4f' )
    if len(stats) > 0:
      np.savetxt( file_root + "_stats.txt", stats, fmt='%0.4f' )
      np.savetxt( file_root + "_sim_stats.txt", sim_stats, fmt='%0.4f' )
    else:
      print "recorder : no stats to save"
    np.savetxt( file_root + "_acceptances.txt", acceptances, fmt='%d' )
    np.savetxt( file_root + "_sims.txt", sims, fmt='%d' )
    if len(invalid_thetas) > 0:
      np.savetxt( file_root + "_invalid_thetas.txt", invalid_thetas, fmt='%0.4f' )
    else:
      print "recorder : no invalid thetas to save"