from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.problems.exponential     import default_params     as load_default_params
from abcpy.algos.mcmc         import abc_mcmc       
from abcpy.algos.rejection         import abc_rejection 
# from abcpy.states.kernel_epsilon import KernelEpsilonState as State
from abcpy.states.kernel_epsilon import KernelEpsilonState as State
from abcpy.states.distance_epsilon import DistanceEpsilonState as RejectState
from abcpy.states.state_recorder       import BaseStateRecorder as Recorder
from abcpy.kernels.gaussian import log_gaussian_kernel
#from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
from abcpy.metropolis_hastings_models.surrogate_metropolis_hastings_model import SurrogateMetropolisHastingsModel as MH_Model

import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda

problem_params = load_default_params()
problem = Problem( problem_params, force_init = True )


# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["obs_statistics"]        = problem.get_obs_statistics()
state_params["theta_prior_rand_func"] = problem.theta_prior_rand
state_params["simulation_function"]   = problem.simulation_function
state_params["statistics_function"]   = problem.statistics_function
state_params["is_marginal"]                = False


theta0 = problem.theta_prior_rand()

Ss = [1,2,10]
nbr_samples  = 50000
epsilons     = [5.0,2.0,1.0,0.1,0.05]
repeats      = 10

  
for S in Ss:

  state_params["S"]  = S
  if state_params["is_marginal"]:
    save_dir = "./uai2014/runs/exponential/abc_mcmc_marginal_s%d/"%(state_params["S"])
  else:
    save_dir = "./uai2014/runs/exponential/abc_mcmc_pseudo_s%d/"%(state_params["S"])
    
  for epsilon in epsilons:
    state_params["epsilon"]                    = epsilon
    for repeat in range(repeats):
      np.random.seed(repeat)
      
      rej_state_params = state_params.copy()
      rej_state_params["S"] = 1
      rej_state = RejectState(theta0, rej_state_params )

      recorder = Recorder(record_stats=False)
      n_reject = 1
    
      rej_thetas, rej_discs = abc_rejection( n_reject, epsilon, rej_state, recorder = recorder  )
      theta0 = rej_thetas[-1]
    
      state  = State( theta0, state_params )
      #recorder = Recorder(record_stats=True)
      #recorder.record_state( state, state.nbr_sim_calls, accepted=False )
    
      epsilon_string = "eps" + str(epsilon).replace(".","p")
      out_name = save_dir + epsilon_string + "_" + "repeat%d"%(repeat+1)
    
      print "***************  RUNNING ABC MCMC ***************"
      print "is_marginal ",state_params["is_marginal"]  
      print "S ",S
      print "epsilon", epsilon
      print "repeat", repeat
      thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, state, recorder  )
    
      print "***************  VIEW RESULTS ***************"
      #problem.view_results( recorder, burnin = 0 )
    
      print "***************  SAVING ******************************"
      recorder.save_results( out_name )
    
      assert len( recorder.get_thetas() ) == nbr_samples + n_reject
      pp.show()
      print "***************  DONE VIEW    ***************"