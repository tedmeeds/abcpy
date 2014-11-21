import numpy as np
import scipy as sp
import pylab as pp
import pdb

def add_to( s, x ):
  if len(s) == 0:
    if len(x.shape) == 1:
      s = np.array([x]).reshape( (1,len(x)))
    else:
      s = x.copy()
  else:
    if len(x.shape) == 1:
      s = np.vstack( (s, x.reshape( (1,len(x))) ))
    else:
      s = np.vstack( (s, x))
  return s
    
class Recorder( object ):
  def __init__(self, state_theta_log_file, state_stats_log_file, \
                    sim_theta_log_file, sim_stats_log_file, \
                    write_each = False ):
    
    # MCMC: the sate is theta and the stats
    self.state_theta_log_file = state_theta_log_file
    self.state_stats_log_file = state_stats_log_file
    
    # SIMS: every theta and stats used in a simulation call
    self.sim_theta_log_file   = sim_theta_log_file
    self.sim_stats_log_file   = sim_stats_log_file
    
    # write_each: depending on speed on simulator, nbr samples, etc, may want to toggle
    # if True, then open file and write at each add
    # if False, only write on finalize
    self.write_each = write_each
    
    # only keep these is we are not directly writing to storage
    if self.write_each is False:
      self.state_thetas  = []
      self.state_stats   = []
      self.sims_thetas   = []
      self.sims_stats    = []
    
    self.sim_calls = []
  
  # return in memory or in storage MCMC thetas 
  def get_thetas( self ):
    if self.write_each is False:
      return self.state_thetas
    else:
      return self.load( self.state_theta_log_file )
   
  # return in memory or in storage MCMC stats 
  def get_stats( self ):
    if self.write_each is False:
      return self.state_stats
    else:
      return self.load( self.state_stats_log_file )
  
  def get_statistics(self):
    return self.get_stats()
     
  # return in memory or in storage MCMC states    
  def get_states( self ):
    if self.write_each is False:
      return self.state_thetas, self.state_stats
    else:
      return self.load( self.state_theta_log_file ), self.load( self.state_stats_log_file )
   
  # return in memory or in storage SIMULATION thetas and stats    
  def get_sims( self ):
    if self.write_each is False:
      return self.sim_thetas, self.sim_stats
    else:
      return self.load( self.sim_theta_log_file ), self.load( self.sim_stats_log_file )
  
  def get_sim_calls(self):
    return self.sim_calls
    
  # add a state of the Markov chain        
  def add_state( self, theta, stats ):
    self.sim_calls.append( 1 )
    if self.write_each:
      self.write( theta, self.state_theta_log_file )
      self.write( stats, self.state_stats_log_file )
    else:
      self.state_thetas = add_to( self.state_thetas, theta )
      self.state_stats  = add_to( self.state_stats, stats )
  
  # add SIMULATION result
  def add_sim( self, theta, stats ):
    if self.write_each:
      self.write( theta, self.sim_theta_log_file )
      self.write( stats, self.sim_stats_log_file )
    else:
      self.sim_thetas = add_to( self.sim_thetas, theta )
      self.sim_stats  = add_to( self.sim_stats, stats )
  
  # in case we are not recording, write to disk at end of run  
  def finalize(self):
    self.sim_calls = np.array(self.sim_calls)
    if self.write_each is False:
      self.write( self.state_thetas, self.state_theta_log_file )
      self.write( self.state_stats, self.state_stats_log_file )
      self.write( self.sims_thetas, self.sim_theta_log_file )
      self.write( self.sims_stats, self.sim_stats_log_file )
  
  # load file if we have defined a filename
  def load( self, filename ):
    if filename is None:
      return None
    else:
      return np.loadtxt( filename )
  
  # write to file if we have a filename    
  def write( self, x, filename ):
    if filename is None:
      return -1
    
    fptr = open( filename, "a+" )
    
    if len(x.shape)==1:
      N=1; D = len(x)
      for d in xrange(D):
        fptr.write( "%0.4f "%(x[d]))
      fptr.write( "\n")
    else:  
      N,D = x.shape
      for n in xrange(N):
        for d in xrange(D):
          fptr.write( "%0.4f "%(x[n,d]))
        fptr.write( "\n")
    fptr.close()
    