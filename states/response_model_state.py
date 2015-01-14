from abcpy.states.kernel_based_state import KernelState
import numpy as np
import pdb

class ResponseModelState(KernelState):
  def __init__( self, params, response_groups = None ):
    super(ResponseModelState, self).__init__( params, response_groups )
    # if response_model is None:
    #   self.response_model = self.params["response_model"]
    # else:
    #   self.response_model = response_model
    
  def new( self, theta, params = None ):
    if theta is None:
      theta = self.theta
    if params is None:
      params = self.params
      
    # response_model will decide if new means copy or just set to same response model (ie for surrogates)
    #response_model = self.response_model.new( self.response_model.params )
    response_groups = [rg.new(rg.params) for rg in self.response_groups]

    s = ResponseModelState( params, response_groups )
    s.set_theta(theta)
    return s
   
  def acquire( self, N = 1 ):
    # run for N more times, do not reset stats already computed
    self.run_simulator_and_compute_statistics( reset = False, S = N )
    
    ngroups = len(self.observation_groups)
    for group_id, sg, rg in zip( range(ngroups), self.observation_groups, self.response_groups ):
      rg.add( self.theta, self.simulation_statistics[-N:,sg.ids], sg.ystar )
      
    #pdb.set_trace()
    self.loglikelihood_is_computed = False
  
  def add( self, thetas, simulation_statistics = None ):
    if simulation_statistics is None:
      simulation_outputs, simulation_statistics = self.run_at_thetas( thetas )
    S,J = simulation_statistics.shape
    ngroups = len(self.observation_groups)
    for group_id, sg, rg in zip( range(ngroups), self.observation_groups, self.response_groups ):
      rg.add( thetas, np.array( [simulation_statistics[:,sg.ids]] ).reshape((S,len(sg.ids))), sg.ystar )
      
    #pdb.set_trace()
    self.loglikelihood_is_computed = False
    
  def loglikelihood(self):
    if self.loglikelihood_is_computed:
      return self.loglikelihood_value
    
    if self.response_groups[0].is_empty():  
      self.run_simulator_and_compute_statistics()
    
      ngroups = len(self.observation_groups)
      for group_id, sg, rg in zip( range(ngroups), self.observation_groups, self.response_groups ):
        rg.add( self.theta, self.simulation_statistics[:,sg.ids], sg.ystar )
        
      #self.response_model.add( self.theta, self.simulation_statistics, self.observation_statistics )
      
    self.compute_loglikelihood()
    
    return self.loglikelihood_value
      
  def compute_loglikelihood(self):
    #pseudo_statistics      = self.simulation_statistics
    #observation_statistics = self.observation_statistics   
    
    #S,J1 = pseudo_statistics.shape
    #N,J  = observation_statistics.shape
    
    #assert J == J1, "observation stats and pseudo stats should be the same"
    
    ngroups = len(self.observation_groups)
    self.loglikelihood_value = 0.0
    for group_id, sg, rg in zip( range(ngroups), self.observation_groups, self.response_groups ):
      #print sg.ystar, rg.loglikelihood( self.theta, sg.ystar )
      self.loglikelihood_value += rg.loglikelihood( self.theta, sg.ystar )
        
    # over all observation statistics
    #loglike_n = self.response_model.loglikelihood( self.theta, observation_statistics )
        
    #self.loglikelihood_value = loglike_n.sum()
      
    self.loglikelihood_is_computed = True
    
  def loglikelihood_rand( self, M=1 ):
    # call likelihood to force running simulator
    if self.response_groups[0].is_empty():   
      self.run_simulator_and_compute_statistics()
      ngroups = len(self.observation_groups)
      for group_id, sg, rg in zip( range(ngroups), self.observation_groups, self.response_groups ):
        rg.add( self.theta, self.simulation_statistics[:,sg.ids], sg.ystar )
      #self.response_model.add( self.theta, self.simulation_statistics, self.observation_statistics )
      
    loglikelihood_rand = np.array([rg.loglikelihood_rand(self.theta, sg.ystar, M) \
                                  for sg,rg in zip(self.observation_groups, self.response_groups)])
  
    return loglikelihood_rand.sum(0)
    
    #return self.response_model.loglikelihood_rand( self.theta, self.observation_statistics, M )
    
  def update( self ):
    ngroups = len(self.observation_groups)
    for group_id, og, rg in zip( range(ngroups), self.observation_groups, self.response_groups ):
      rg.update( og )