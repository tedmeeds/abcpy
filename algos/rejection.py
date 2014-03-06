import numpy as np
import pylab as pp

def abc_rejection( nbr_samples, epsilon, state, StateClass, all_states = None, verbose = True ):
  # required functions

    
  # init states
  thetas           = []
  nbr_sim_calls    = 0
  acceptances = []
  nbr_accepts = 0
  for n in xrange(nbr_samples):
    if verbose:
      print "n = %8d of %d"%(n,nbr_samples)
    accepted = False
    
    # repeat until accepted
    while accepted is False:
      # sample parameter setting from (prior) function
      theta = state.theta_prior_rand()
      
      # simuation -> outputs -> statistics -> discrepancies
      theta_state    = StateClass( theta, state.params )
      theta_disc     = theta_state.discrepancy()
      
      #if verbose:
      #  print "    theta = %3.3f   stats = %3.3f  disc = %3.3f"%(theta, theta_state.get_statistics(), theta_disc )
      
      # all discrepancies much be less than all epsilons
      if np.all( theta_disc <= epsilon ):
        accepted = True
        thetas.append(theta)
        nbr_accepts += 1
        acceptances.append(1)
      
      if all_states is not None:  
        all_states.add( theta_state, accepted )
  #print "here"      
  # package results
  #print thetas
  thetas = np.array(thetas) 
  print thetas
  #print "here" 
  acceptances = np.array(acceptances)   
  #print "here"         
  return thetas
  
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
  
  
      
  