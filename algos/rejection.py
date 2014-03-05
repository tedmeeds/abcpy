import numpy as np
import pylab as pp

def abc_rejection( N, params ):
  # required functions
  obs_stats        = params["obs_stats"]
  rand_func        = params["rand_func"]
  sim_func         = params["sim_func"]
  stats_func       = params["stats_func"]
  discrepancy_func = params["discrepancy_func"]
  epsilon          = params["epsilon"]
  
  # extra results to keep
  keep_discrepancy = False
  keep_stats       = False
  keep_outputs     = False
  
  if params.has_key( "keep_discrepancy"):
    keep_discrepancy = params["keep_discrepancy"]
  if params.has_key( "keep_stats"):
    keep_stats = params["keep_stats"]
  if params.has_key( "keep_outputs"):
    keep_outputs = params["keep_outputs"]
  if params.has_key( "keep_rejections"):
    keep_rejections = params["keep_rejections"]
    
  # init states
  X           = []
  D           = []
  S           = []
  O           = []
  R           = []
  Rstats      = []
  nbr_sim_calls   = 0
  acceptances = []
  nbr_accepts = 0
  for n in xrange(N):
    
    accepted = False
    
    # repeat until accepted
    while accepted is False:
      # sample parameter setting from (prior) function
      x = rand_func()
      
      # simuation -> outputs -> statistics -> discrepancies
      x_sim_outs = sim_func( x ); nbr_sim_calls+=1
      x_stats    = stats_func( x_sim_outs )
      x_disc     = discrepancy_func( x_stats, obs_stats )
      
      # all discrepancies much be less than all epsilons
      if np.all( x_disc <= epsilon ):
        accepted = True
        X.append(x)
        nbr_accepts += 1
        acceptances.append(1)
        
        if keep_outputs:
          O.append( x_sim_outs )
        if keep_stats:
          S.append( x_stats )
        if keep_discrepancy:
          D.append( x_disc )
      else:
        acceptances.append(0)
        if keep_rejections:
          R.append( x )
          Rstats.append( x_stats )
        
  # package results
  X = np.array(X) 
  acceptances = np.array(acceptances)       
  if keep_outputs:
    O = np.array(O)
  if keep_stats:
    S = np.array(S)
  if keep_discrepancy:
    D = np.array(D)
  if keep_rejections:
    R = np.array(R)
    Rstats = np.array(Rstats)
  outputs = { \
               "nbr_sim_calls" : nbr_sim_calls, \
               "nbr_accepts"   : nbr_accepts, \
               "accept_rate"   : float(nbr_accepts)/float(nbr_sim_calls), \
               "acceptances"   : acceptances, \
               "X"             : X, \
               "RHO"           : D, \
               "STATS"         : S, \
               "OUTS"          : O, \
               "REJECT_X"      : R, \
               "REJECT_S"      : Rstats, \
            }
            
  return X, outputs
  
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
  
  
      
  