import numpy as np
import pdb

class ABC_State(object):
  def __init__( self, params, response_groups = None ):
    # TODO: changed and removed theta from init
    self.theta         = None
    self.params        = params
    self.nbr_sim_calls = 0
    self.nbr_sim_calls_this_iter = 0
    self.simulation_outputs     = []
    self.simulation_statistics  = []
    
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
  
    if len(self.observation_statistics.shape) > 1:
      self.N, self.J = self.observation_statistics.shape
    else:
      self.N = 1
      self.J = len(self.observation_statistics)
      self.observation_statistics = self.observation_statistics.reshape( (self.N,self.J))
    
  def set_theta( self, theta ):
    self.theta = theta
    
  def update_post_mh(self):
    raise NotImplementedError
      
  def add_sim_call( self, N=1 ):
    self.nbr_sim_calls           += N
    self.nbr_sim_calls_this_iter += N
  
  def reset_nbr_sim_calls_this_iter(self):
    self.nbr_sim_calls_this_iter = 0
    
  def loglikelihood( self ):
    raise NotImplementedError
    
  def loglikelihood_rand( self, M=1 ):
    raise NotImplementedError
    
  def run_at_thetas( self, thetas ):
    
    simulation_outputs     = []
    simulation_statistics  = []
    S = len(thetas)
    for theta in thetas:
      # simulation -> outputs -> statistics
      simulation_outputs.append( np.squeeze( self.simulation_function( theta ) ) )
      
      if self.D is None:
        #pdb.set_trace()
        if simulation_outputs[-1].__class__ == np.ndarray:
          #self.D = len(self.simulation_outputs[-1])
          self.D = simulation_outputs[-1].size
        else:
          self.D = 1
        
      # keep track of simulation calls
      self.add_sim_call()
    
      # process for statistics
      simulation_statistics.append( self.statistics_function( simulation_outputs[-1] ) )
      
    # make into arrays
    simulation_outputs    = np.array(simulation_outputs).reshape( (S,self.D))
    simulation_statistics = np.array(simulation_statistics).reshape( ( S, self.J ) )
    
    return simulation_outputs, simulation_statistics
      
  def run_simulator_and_compute_statistics(self, reset = True, S = None ):
    if S is None:
      S = self.S
      
    # keep track of old results (eg for acquisition functions -- adding new design pts)
    if reset is False:
      old_sim_outputs  = self.simulation_outputs
      old_sim_stats    = self.simulation_statistics
      
    # resetting outputs and statistics (save these before if necessary)
    self.simulation_outputs     = []
    self.simulation_statistics  = []
    
    # sometimes there may be more that one simulation run, this must be set in params
    for s in range(S):
      # simulation -> outputs -> statistics
      outs = self.simulation_function( self.theta ) 
      c = 0
      while outs is None:
        c+=1
        print "SIMULATOR RETURNED NONE!!"
        if c > 9:
          assert False, "Something wrong at these parameter settings"
        outs = self.simulation_function( self.theta )
        
      self.simulation_outputs.append( np.squeeze( outs ) )
      
      
      
      if self.D is None:
        #pdb.set_trace()
        if self.simulation_outputs[-1].__class__ == np.ndarray:
          #self.D = len(self.simulation_outputs[-1])
          self.D = self.simulation_outputs[-1].size
        else:
          self.D = 1
          
      # keep track of simulation calls
      self.add_sim_call()
      
      # process for statistics
      self.simulation_statistics.append( self.statistics_function( self.simulation_outputs[-1] ) )
      
    # make into arrays
    self.simulation_outputs    = np.array(self.simulation_outputs).reshape( (S,self.D))
    self.simulation_statistics = np.array(self.simulation_statistics).reshape( ( S, self.J ) )
    
    #print "self.simulation_statistics: ",self.simulation_statistics
    if reset is False:
      if len(old_sim_outputs) > 0:
        self.simulation_outputs    = np.vstack( (old_sim_outputs,self.simulation_outputs))
        self.simulation_statistics = np.vstack( (old_sim_stats,self.simulation_statistics))
        