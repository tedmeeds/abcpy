from simulation_response_kernel import SimulationResponseKernel
from abcpy.helpers import gaussian_logpdf, heavyside

import numpy as np
import pylab as pp
import pdb

class EpsilonHeavysideExponentialResponseKernel( SimulationResponseKernel ):
    
  def load_params( self, params ):
    self.down = 1
    self.up   = 0
    
    if params.has_key("epsilon"):
      self.epsilon = params["epsilon"]
      
    if params.has_key("min_epsilon"):
      self.min_epsilon = params["min_epsilon"] 
    else:
      self.min_epsilon = self.epsilon.copy()
   
      
    if params.has_key("cliff_at"):
      self.cliff_at = params["cliff_at"] 
    else:
      self.cliff_at = None
        
    # force user to decide if down-side epsilon or up-side epsilon 
    if params["direction"] == "down":
      self.direction = self.down
    elif params["direction"] == "up":
      self.direction = self.up
    else:
      self.direction = self.down
      
    self.epsilon_update_rule = None
    if params.has_key("epsilon_update_rule"):
      self.epsilon_update_rule = params["epsilon_update_rule"]
      self.epsilon_update_params = params["epsilon_update_params"]
      self.ema_decay = self.epsilon_update_params["decay"]
      
      if self.epsilon_update_params.has_key("ema"):
        self.ema     = self.epsilon_update_params["ema"]
        self.ema_var = self.epsilon_update_params["ema_var"]
        pdb.set_trace()
      else:
        self.ema     = self.epsilon
        self.ema_var = self.epsilon
      self.quantile = self.epsilon_update_params["quantile"]
      
      #pdb.set_trace()
      
  def update_post_mh( self, observation_group, simulation_statistics, params ):
    #print "before", self.ema, self.epsilon
    if self.epsilon_update_rule is not None:
      for ss in simulation_statistics:
        diff = self.discrepancy( observation_group.ystar, ss ) - self.ema
        incr = self.ema_decay*diff
        self.ema += self.ema_decay*diff
        self.ema_var = (1.0 - self.ema_decay) * (self.ema_var + diff * incr)
      
      epsilon = self.quantile*self.ema 
      for i in range(len(self.epsilon)):
        oo=epsilon.copy()
        pop=self.epsilon.copy()
        self.epsilon[i] = max( epsilon[i], self.min_epsilon[i])
        
        if self.epsilon[i] < self.min_epsilon[i]:
          pdb.set_trace()
      print "self.epsilon  ", self.epsilon, self.ema
  
  def discrepancy( self, ystar, y ):  
    J = len(y)
    d = y - ystar
    
    if self.direction == self.up:
      d *= -1
      
    N = 1
    # assume every observation is outside of tube
    disc = np.zeros( J )
  
    for j in range(J):
      h = heavyside( d[j] )
      if h > 0.5:
        disc[j] = d[j]
    return disc
        
  def loglikelihood( self, observation_statistics, pseudo_statistics ):
    # sh = observation_statistics.shape
    # if len(sh) > 1:
    #   N,J = sh
    # elif len(sh) == 0:
    #   N=1
    #   J=1
    # else:
    #   N = sh[0]
    #   J = 1
    # 
    # psh = pseudo_statistics.shape
    # if len(psh) > 1:
    #   S,J = psh
    # else:
    #   S = psh[0]
    #   J = 1
    # 
    # # if S>1:
    # #   pdb.set_trace()
    # if N > 1:
    #   assert S == 1,"only allow one greater than the other"
    # elif S > 1:
    #   N=S
          
    # start with the difference d =  y - y_star
    J = len(pseudo_statistics)
    d = pseudo_statistics - observation_statistics
    
    if self.direction == self.up:
      d *= -1
      
    N = 1
    # assume every observation is outside of tube
    loglikelihood = np.zeros( J )
  
    for j in range(J):
      h = heavyside( d[j] )
      if h > 0.5:
        loglikelihood[j] = -d[j]/self.epsilon[j]

      
      if self.cliff_at is not None:
        if pseudo_statistics[j] == self.cliff_at[j]:
          loglikelihood[j] = -np.inf
    #pdb.set_trace()
    return np.sum(loglikelihood)