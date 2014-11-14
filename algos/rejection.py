import numpy as np
import pylab as pp
import pdb

def abc_rejection( nbr_samples, lower_epsilon, upper_epsilon, state, prior_rand, recorder= None, verbose = True ):
  # required functions

    
  # init states
  discs = []
  thetas           = []
  nbr_sim_calls    = 0
  acceptances = []
  nbr_accepts = 0
  for n in xrange(nbr_samples):
    if verbose:
      print "n = %8d of %d"%(n,nbr_samples)
    accepted = False
    this_iters_sim_calls = 0
    
    # repeat until accepted
    while accepted is False:
      # sample parameter setting from (prior) function
      theta = prior_rand()
      
      # simuation -> outputs -> statistics -> discrepancies
      theta_state    = state.new( theta, state.params )
      theta_disc     = theta_state.discrepancy()
      this_iters_sim_calls += theta_state.nbr_sim_calls_this_iter
      
      # all discrepancies much be less than all epsilons
      if np.all( (theta_disc <= upper_epsilon)&(theta_disc >= lower_epsilon) ):
        accepted = True
        nbr_accepts += 1
        thetas.append(theta)
        nbr_sim_calls += this_iters_sim_calls
        discs.append(np.sum(theta_disc))
      
      # this should work even if do not use "all_states"
      if recorder is not None:  
        # only accepted are valid states for rejection sampling
        if accepted:
          recorder.add_state( theta_state.theta, theta_state.simulation_statistics )
          #recorder.record_state( theta_state, this_iters_sim_calls, accepted )
          
        # we may care about the rejected samples, however
        #else:
        #  recorder.record_invalid( theta_state )
          
  thetas = np.array(thetas) 
  discs = np.array(discs)
  recorder.finalize()
  #acceptances = np.array(acceptances)   
  return thetas, discs
  
if __name__ == "__main__":
  assert False, "This demo will no longer work, see /examples/ for working demos."
  
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
  
  
      
  