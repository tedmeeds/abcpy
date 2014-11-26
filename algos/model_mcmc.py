import numpy as np
import pylab as pp

  
def abc_mcmc( nbr_samples, model, verbose = False, verbose_rate = 100 ):
  #assert state is not None, "need to start with a state"
  
  # init with current state's theta
  #theta_loglik   = state.loglikelihood()
  #theta_logprior = state.logprior()
  loglik         = model.log_posterior() #theta_loglik + theta_logprior
  
  # init states
  thetas          = [model.current.theta]
  LL              = [loglik]
  nbr_sim_calls   = model.current.nbr_sim_calls
  acceptances     = [True]
  sim_calls       = [model.current.nbr_sim_calls]
  nbr_accepts     = 1
  model.stay_in_current_state()
  
  for n in xrange(nbr_samples):
    if verbose and np.mod(n+1,verbose_rate)==0:
      print "T = ", n+1, " of ",nbr_samples, " loglik: ", loglik
    model.reset_nbr_sim_calls_this_iter()
    this_iters_sim_calls = 0
    
    # create new state for proposal q
    model.propose_state()
    model.update_current()  # create new state is pseudo-marginal
    
    #
    
    
    

    #model.set_proposed_state( q_state )
    #model.set_current_state( state )
    log_acc = model.log_acceptance()
    
    if verbose and verbose_rate == 1:
       print "  current  loglik: ", model.current.loglikelihood()
       print "  proposed loglik: ", model.proposed.loglikelihood()
    # keep track of all sim calls
    #this_iters_sim_calls += q_state.nbr_sim_calls
    
    # only count if "marginal"; peseduo-marginal does not run simulations
    #this_iters_sim_calls += int(state.is_marginal)*state.nbr_sim_calls
    
    # can also send as u-stream
    u = np.random.rand() 
  
    # the MH accept-reject step
    accepted = False   
    if (log_acc >= 0) or (u <= np.exp( log_acc ) ):
      accepted = True
      nbr_accepts += 1
      
      # move to new state
      model.move_to_proposed_state()
      #state     = q_state
      #loglik    = state.loglikelihood() + state.logprior()
    else:
      model.stay_in_current_state()

    
    # keep track of all states in chain
    #if all_states is not None:
    #  all_states.add( state, this_iters_sim_calls, accepted )
    
    loglik = model.log_posterior()
    model.recorder.add_state( model.current.theta, model.current.simulation_statistics.mean(0).reshape( (1,model.current.simulation_statistics.shape[1]) ), loglik )
    
    nbr_sim_calls += this_iters_sim_calls
    
    acceptances.append( accepted )
    accept_rate     = float(nbr_accepts)/float(n+1)
    efficiency_rate = 0.0
    if nbr_sim_calls > 0:
      efficiency_rate = float(nbr_accepts)/float(nbr_sim_calls)
    sim_calls.append( this_iters_sim_calls )
    LL.append(loglik)
    thetas.append(model.current.theta.copy())
    
  acceptances = np.array(acceptances)
  sim_calls   = np.array(sim_calls)
  thetas      = np.array(thetas)
  LL          =  np.array(LL)  
  model.recorder.finalize()
  return thetas, LL, acceptances,sim_calls
  
# if __name__ == "__main__":
#   def rand_func():
#     return 3*np.random.randn()
#     
#   def sim_func( x ):
#     return x + np.random.randn(3)
#     
#   def stats_func( outputs ):
#     return np.mean(outputs)
#     
#   def discrepancy_func( x_stats, obs_stats ):
#     return np.abs( x_stats - obs_stats )
#   
#   N = 10000
#   epsilon = 1.1
#   x_star = 2.0
#   x_star_outs = sim_func( x_star )
#   x_star_stats = stats_func( x_star_outs )
#     
#   params = {}
#   params["obs_stats"]        = x_star_stats
#   params["rand_func"]        = rand_func
#   params["sim_func"]         = sim_func
#   params["stats_func"]       = stats_func
#   params["discrepancy_func"] = discrepancy_func
#   params["epsilon"]          = epsilon
#   params["keep_discrepancy"] = True
#   params["keep_stats"]       = True
#   params["keep_outputs"]     = True
#   params["keep_rejections"]  = True
#   
#   X, results = abc_rejection( N, params )
#   
#   pp.figure(1)
#   pp.clf()
#   pp.subplot(2,2,1)
#   pp.plot( results["REJECT_X"], results["REJECT_S"], 'ro', alpha = 0.25 )
#   pp.plot( results["X"], results["STATS"], 'go', alpha = 0.85 )
#   ax = pp.axis()
#   pp.vlines( [x_star], ax[2], ax[3], color = "k", linewidths=3)
#   pp.hlines( x_star_stats, ax[2], ax[3], color = "k", linewidths=3)
#   pp.fill_between( np.linspace(ax[0], ax[1],10), x_star_stats-epsilon, x_star_stats+epsilon, color="g", alpha=0.5 )
#   pp.ylabel( "discrepancy")
#   pp.xlabel( "theta")
#   pp.axis(ax)
# 
#   pp.subplot(2,2,3)
#   pp.hist( X, 10, normed=True, alpha=0.5, color="g" )
#   ax2 = pp.axis()
#   pp.vlines( [x_star], ax[2], ax[3], color = "k", linewidths=3)
#   pp.axis( [ax[0],ax[1],ax2[2],ax2[3]] )
# 
#   pp.ylabel( "P(theta)")
#   pp.xlabel( "theta")
#   
#   pp.subplot(2,2,2)
#   pp.hist( results["STATS"], 10, normed=True, alpha=0.5, color="g", orientation = 'horizontal' )
#   ax2 = pp.axis()
#   pp.hlines( x_star_stats, ax2[2], ax2[3], color = "k", linewidths=3)
#   pp.axis( [ax2[0], ax2[1], ax[2], ax[3]] )
# 
#   pp.ylabel( "stats")
#   pp.xlabel( "P(stats)")
#   
#   pp.show()
#   
#   
      
  