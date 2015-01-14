from abcpy.factories import *
from abcpy.problems.blowfly.blowfly import *

import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = default_params()
problem_params["epsilon"] = 0.5
problem_params["q_factor"] = 0.2
problem = BlowflyProblem( problem_params, force_init = True )


# # since we are running abc_rejection, use a distance epsilon state
# state_params = {}
# state_params["S"]                          = 1
# state_params["obs_statistics"]             = problem.get_obs_statistics()
# state_params["theta_prior_rand_func"]      = problem.theta_prior_rand
# state_params["theta_prior_logpdf_func"]    = problem.theta_prior_logpdf
# state_params["theta_proposal_rand_func"]   = problem.theta_proposal_rand
# state_params["theta_proposal_logpdf_func"] = problem.theta_proposal_logpdf
# state_params["simulation_function"]        = problem.simulation_function
# state_params["statistics_function"]        = problem.statistics_function
# state_params["log_kernel_func"]            = log_gaussian_kernel
# state_params["is_marginal"]                = True
# state_params["epsilon"]                    = 1.0 #np.array([0.1,0.1,0.1,1.0])

nbr_samples = 10000
#epsilon     = 0.5
state_params = state_params_factory.scrape_params_from_problem( problem, S = 1 )
mcmc_params  = mcmc_params_factory.scrape_params_from_problem( problem, type="mh", is_marginal = True, nbr_samples = nbr_samples )
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