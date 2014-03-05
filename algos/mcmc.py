import numpy as np
import pylab as pp

class SimpleState( object ):
  def __init__( self, x, params ):
    self.obs_stats        = params["obs_stats"]
    self.q_func           = params["q_func"]
    self.sim_func         = params["sim_func"]
    self.stats_func       = params["stats_func"]
    self.log_kernel_func  = params["log_kernel_func"]
    self.epsilon          = params["epsilon"]
    self.S                = params["S"]
    self.is_marginal      = params["is_marginal"]
    self.x                = x
    self.nbr_sim_calls    = 0
    self.loglik           = None
    
  def loglikelihood( self ):
    if self.loglik is not None:
      return self.loglik
      
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
  
def abc_mcmc( all_states, params ):
  state = all_states[-1]
  x     = state.x
  
  NewState     = params["state_class"]
  state_params = params["state_params"]
  
  # init states
  X           = []
  LL          = []
  nbr_sim_calls   = 0
  acceptances = []
  nbr_accepts = 0
  for n in xrange(N):
    
    # sample q from a proposal distribution
    q = q_func( x )
    
    # create new state for proposal q
    q_state    = NewState( q, state_params )
    
    # simulation -> outputs -> statistics -> loglikelihood 
    q_loglik   = q_state.loglikelihood()
    
    # prior log-density
    q_logprior = q_state.logprior( q )
    
    # keep track of all sim calls
    nbr_sim_calls += q_state.nbr_sim_calls
    
    # for marginal sampler, we need to re-run the simulation at the current location
    if is_marginal or state is None:
      state = NewState( x, state_params )

    # likelihood only computed once
    x_loglik   = state.loglikelihood()
    x_logprior = state.logprior( x )
    
    # this is ok for marginal and pseudo-marginal -- only counted once per instantiation
    nbr_sim_calls += x_state.nbr_sim_calls  
      
    # log-density of proposals
    q_to_x_logproposal = q_state.logproposal( x, q )  
    x_to_q_logproposal =  state.logproposal( q, x ) 
    
    # Metropolis-Hastings acceptance log-probability and probability
    log_acc = min(0.0, q_loglik - x_loglik + q_logprior - x_logprior + x_to_q_logproposal - q_to_x_logproposal )
    acc     = np.exp( log_acc )
    
    # can also send as u-stream
    u = np.random.rand() 
  
    # the MH accept-reject step
    accepted = False
    if u <= acc:
      accepted = True
      
      # move to new state
      x     = q.copy()
      state = q_state.copy()
      loglik = q_loglik + q_logprior
      
      # keep track of all states in chain
      all_states.add( accepted, state )
    else:
      # also keep track of rejecttted states (may be of interest for some models)
      all_states.add( accepted, state, q_state )
      loglik = p_loglik + p_logprior
    
    LL.append(loglik)
    X.append(x.copy())
  
  X  = np.array(X)
  LL = np.array(LL)  
  return X, all_states
  
if __name__ == "__main__":
  def rand_func():
    return 3*np.random.randn()
    
  def sim_func( x ):
    return x + np.random.randn(3)
    
  def stats_func( outputs ):
    return np.mean(outputs)
    
  def discrepancy_func( x_stats, obs_stats ):
    return np.abs( x_stats - obs_stats )
  
  N = 10000
  epsilon = 1.1
  x_star = 2.0
  x_star_outs = sim_func( x_star )
  x_star_stats = stats_func( x_star_outs )
    
  params = {}
  params["obs_stats"]        = x_star_stats
  params["rand_func"]        = rand_func
  params["sim_func"]         = sim_func
  params["stats_func"]       = stats_func
  params["discrepancy_func"] = discrepancy_func
  params["epsilon"]          = epsilon
  params["keep_discrepancy"] = True
  params["keep_stats"]       = True
  params["keep_outputs"]     = True
  params["keep_rejections"]  = True
  
  X, results = abc_rejection( N, params )
  
  pp.figure(1)
  pp.clf()
  pp.subplot(2,2,1)
  pp.plot( results["REJECT_X"], results["REJECT_S"], 'ro', alpha = 0.25 )
  pp.plot( results["X"], results["STATS"], 'go', alpha = 0.85 )
  ax = pp.axis()
  pp.vlines( [x_star], ax[2], ax[3], color = "k", linewidths=3)
  pp.hlines( x_star_stats, ax[2], ax[3], color = "k", linewidths=3)
  pp.fill_between( np.linspace(ax[0], ax[1],10), x_star_stats-epsilon, x_star_stats+epsilon, color="g", alpha=0.5 )
  pp.ylabel( "discrepancy")
  pp.xlabel( "theta")
  pp.axis(ax)

  pp.subplot(2,2,3)
  pp.hist( X, 10, normed=True, alpha=0.5, color="g" )
  ax2 = pp.axis()
  pp.vlines( [x_star], ax[2], ax[3], color = "k", linewidths=3)
  pp.axis( [ax[0],ax[1],ax2[2],ax2[3]] )

  pp.ylabel( "P(theta)")
  pp.xlabel( "theta")
  
  pp.subplot(2,2,2)
  pp.hist( results["STATS"], 10, normed=True, alpha=0.5, color="g", orientation = 'horizontal' )
  ax2 = pp.axis()
  pp.hlines( x_star_stats, ax2[2], ax2[3], color = "k", linewidths=3)
  pp.axis( [ax2[0], ax2[1], ax[2], ax[3]] )

  pp.ylabel( "stats")
  pp.xlabel( "P(stats)")
  
  pp.show()
  
  
      
  