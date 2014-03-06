from abcpy.state import BaseState

# =========================================================================== #
#
#  KernelEpsilonState: computes likelihood based on average of S kernel calls.
#                      Usage: kernel-abc-mcmc
#
# =========================================================================== #
class KernelEpsilonState( BaseState ):
  
  def load_params( self, params ):
    self.obs_stats        = params["obs_stats"]
    self.q_func           = params["q_func"]
    self.sim_func         = params["sim_func"]
    self.stats_func       = params["stats_func"]
    self.log_kernel_func  = params["log_kernel_func"]
    self.epsilon          = params["epsilon"]
    self.S                = params["S"]
    self.is_marginal      = params["is_marginal"]
    self.loglik           = None
    
  def loglikelihood( self, x = None ):
    if (x is None) and (self.loglik is not None):
      return self.loglik
    
    # note we are changing the state here, not merely calling a function
    if x is not None:
      self.x = x
    
    # approximate likelihood by average over S simulations    
    self.loglikelihoods = np.zeros(self.S)
    self.sim_outs       = []
    self.stats          = []
    self.loglik         = []
    for s in range(self.S):
      # simulation -> outputs -> statistics -> loglikelihood
      self.sim_outs.append( self.sim_func( self.x ) ); self.nbr_sim_calls+=1
      self.stats.append( self.stats_func( self.sim_outs[-1] ) )
      self.loglikelihoods[s] = log_kernel_func( self.stats[-1], self.obs_stats, self.epsilon )
    self.loglik = logsumexp( self.loglikelihoods )
    
    return self.loglik