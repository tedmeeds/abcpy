import numpy as np
import pdb
# -- subset of statistics canbe treated and modeled differently.  
# -- some observations might be iids vectors
# -- some observations may be sets of constraints

class ObservationGroup( object ):
  def __init__( self, stat_ids, observation_statistics, params ):
    self.ids = stat_ids # which statistics are used for this group
    self.observation_statistics = observation_statistics # subset of all observations
    self.ystar = observation_statistics
    self.params = params
    if params.has_key("update_ystar_down"):
      self.ema = self.ystar.copy()
      self.ema_var = 0.0
      self.ema_decay = params["update_ystar_down"]
      self.quantile = 0.0
      
  def update_post_mh( self, response_group, simulation_statistics, params ):
    
    if self.params.has_key("update_ystar_down"):
      for ss in simulation_statistics:
        diff = ss - self.ema
        incr = self.ema_decay*diff
        self.ema += self.ema_decay*diff
        self.ema_var = (1.0 - self.ema_decay) * (self.ema_var + diff * incr)
    
      self.ystar = min( self.ystar, self.ema - self.quantile*np.sqrt(self.ema_var) )
      self.observation_statistics = self.ystar
      print "new ystar: ", self.ystar, self.ema #, np.sqrt(self.ema_var), response_group.params["epsilon"]