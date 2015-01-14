from abcpy.factories import *
from abcpy.problems.blowfly.blowfly import *

import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = default_params()
problem_params["epsilon"] = 0.5
problem_params["q_factor"] = 0.1
problem = BlowflyProblem( problem_params, force_init = True )

# model_params = {}
# # adaptive-SL params
# model_params["xi"]            = 0.1
# model_params["M"]             = 10
# model_params["deltaS"]        = 20
# model_params["max_nbr_tries"] = 10
# model = MH_Model( model_params)

# # since we are running abc_rejection, use a distance epsilon state
# state_params = {}
# state_params["S"]                          = 2
# state_params["obs_statistics"]             = problem.get_obs_statistics()
# state_params["theta_prior_rand_func"]      = problem.theta_prior_rand
# state_params["theta_prior_logpdf_func"]    = problem.theta_prior_logpdf
# state_params["theta_proposal_rand_func"]   = problem.theta_proposal_rand
# state_params["theta_proposal_logpdf_func"] = problem.theta_proposal_logpdf
# state_params["simulation_function"]        = problem.simulation_function
# state_params["statistics_function"]        = problem.statistics_function
# state_params["zero_cross_terms"]           = False # change this if we want diagonal estimates
# #state_params["log_kernel_func"]            = log_gaussian_kernel
# state_params["is_marginal"]                = False
# state_params["epsilon"]                    = 0.5 #np.array([0.1,0.1,0.1,1.0])
# state_params["hierarchy_type"]             = "jeffreys"
# state_params["hierarchy_type"]             = "just_gaussian"

nbr_samples = 200
#reject_epsilon = 7.0

# rej_state_params = state_params.copy()
# rej_state_params["S"] = 1
# rej_state = RejectState(None, rej_state_params )
# recorder = Recorder(record_stats=True)
# n_reject = 1
#print "***************  RUNNING REJECTION ***************"
#rej_thetas, rej_discs = abc_rejection( n_reject, reject_epsilon, rej_state, recorder = recorder  )
#theta0 = rej_thetas[-1]
#theta0 = np.array([ 3.07687576, -0.86457619,  6.1387475 , -3.85783667, -0.40429067,  9.        ])
    
nbr_samples = 10000
#epsilon     = 0.5
state_params = state_params_factory.scrape_params_from_problem( problem, S = 20 )
state_params["diagonalize"] = False
mcmc_params  = mcmc_params_factory.scrape_params_from_problem( problem, type="mh", is_marginal = True, nbr_samples = nbr_samples )
algo_params = { "modeling_approach"  : "local_model",
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
print "***************  RUNNING SL MCMC ***************"
thetas, LL, acceptances,sim_calls = algo( nbr_samples, \
                                          model, \
                                          verbose = verbose, \
                                          verbose_rate = 10  )
print " ACCEPT RATE = %0.3f"%(acceptances.mean())
print "***************  DONE SL MCMC    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = nbr_samples/2 )
pp.show()
print "***************  DONE VIEW    ***************"