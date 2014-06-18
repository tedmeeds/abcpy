from abcpy.response_kernel import SimulationResponseKernel
from abcpy.helpers import gaussian_logpdf, heavyside

import numpy as np
import pylab as pp
import pdb

class EpsilonQuadraticIntervalResponseKernel( SimulationResponseKernel ):
    
  def load_params( self, params ):
    self.intervals = params["intervals"]
    self.lower_intervals = self.intervals[0]
    self.upper_intervals = self.intervals[1]
    
    if params.has_key("epsilon"):
      self.epsilon = params["epsilon"]
      
    if params.has_key("min_epsilon"):
      self.min_epsilon = params["min_epsilon"] 
    else:
      self.min_epsilon = self.epsilon.copy()
      
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
      
  def update_post_mh( self, observation_group, simulation_statistics, params ):
    #print "before", self.ema, self.epsilon
    if self.epsilon_update_rule is not None:
      for ss in simulation_statistics:
        diff = self.discrepancy( observation_group.ystar, ss ) - self.ema
        incr = self.ema_decay*diff
        self.ema += self.ema_decay*diff
        self.ema_var = (1.0 - self.ema_decay) * (self.ema_var + diff * incr)
      epsilon = self.quantile*self.ema #np.sqrt(self.ema_var) #- np.sqrt(self.ema_var)
      for i in range(len(self.epsilon)):
        oo=epsilon.copy()
        pop=self.epsilon.copy()
        self.epsilon[i] = max( epsilon[i], self.min_epsilon[i])
        if self.epsilon[i] < self.min_epsilon[i]:
          pdb.set_trace()
      print "self.epsilon  ", self.epsilon, self.ema
      
  def discrepancy( self, ystar, y ):  
    J = len(y)
    disc = np.zeros( J )
  
    for j in range(J):
      
      if y[j] < self.lower_intervals[j]:
        disc[j] = self.lower_intervals[j] - y[j]
      elif y[j] > self.upper_intervals[j]:
        disc[j] = y[j] - self.upper_intervals[j]
    return disc    
    
  def loglikelihood( self, observation_statistics, pseudo_statistics ):
    J = len(pseudo_statistics)
    d = pseudo_statistics - observation_statistics
    
    # assume every observation is outside of tube
    loglikelihood = np.zeros( J )
    disc = self.discrepancy( observation_statistics, pseudo_statistics )
    for j in range(J):
      if disc[j]>0:
        loglikelihood[j] = -0.5*pow( disc[j]/self.epsilon[j], 2 )
    
    return np.sum(loglikelihood)