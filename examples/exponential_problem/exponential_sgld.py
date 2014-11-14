from abcpy.factories import *
from abcpy.problems.exponential     import *

import pylab as pp
import numpy as np

problem_params = default_params()
problem_params["epsilon"] = 0.1
problem = ExponentialProblem( problem_params, force_init = True )

nbr_samples = 10000
state_params = state_params_factory.scrape_params_from_problem( problem, S = 5 )
mcmc_params  = mcmc_params_factory.scrape_params_from_problem( problem, type="sgldmh", is_marginal = False, nbr_samples = nbr_samples )
algo_params = { "modeling_approach"  : "kernel",
                "observation_groups" : problem.get_obs_groups(),
                "state_params"       : state_params,
                "mcmc_params"        : mcmc_params,
                "algorithm"          : "model_mcmc"
              }
recorder_params = {}  
algo, model, state  = algo_factory.create_algo_and_state( algo_params )
recorder     = recorder_factory.create_recorder( recorder_params )

state.theta = problem.theta_prior_rand()
model.set_current_state( state )
model.set_recorder( recorder )

verbose = True
print "***************  RUNNING MODEL ABC MCMC ***************"
thetas, LL, acceptances,sim_calls = algo( nbr_samples, \
                                          model, \
                                          verbose = verbose, \
                                          verbose_rate = 100  )
print " ACCEPT RATE = %0.3f"%(acceptances.mean())
print "***************  DONE ABC MCMC    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = nbr_samples/2 )
pp.show()
print "***************  DONE VIEW    ***************"