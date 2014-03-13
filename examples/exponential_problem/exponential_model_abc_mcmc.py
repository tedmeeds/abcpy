from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.problems.exponential     import default_params     as load_default_params
from abcpy.algos.model_mcmc         import abc_mcmc       
# from abcpy.states.kernel_epsilon import KernelEpsilonState as State
from abcpy.states.synthetic_likelihood import SyntheticLikelihoodState as State
from abcpy.states.state_recorder       import BaseStateRecorder as Recorder
from abcpy.kernels.gaussian import log_gaussian_kernel
#from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as Model
from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel as Model

import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
problem = Problem( problem_params, force_init = True )


# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                          = 2
state_params["obs_statistics"]             = problem.get_obs_statistics()
state_params["theta_prior_rand_func"]      = problem.theta_prior_rand
state_params["theta_prior_logpdf_func"]    = problem.theta_prior_logpdf
state_params["theta_proposal_rand_func"]   = problem.theta_proposal_rand
state_params["theta_proposal_logpdf_func"] = problem.theta_proposal_logpdf
state_params["simulation_function"]        = problem.simulation_function
state_params["statistics_function"]        = problem.statistics_function
state_params["log_kernel_func"]            = log_gaussian_kernel
state_params["is_marginal"]                = False
state_params["epsilon"]                    = 0.0
#state_params["hierarchy_type"]      = "jeffreys"
#state_params["hierarchy_type"]      = "jeffreys"



model_params = {}
# adaptive-SL params
model_params["xi"]            = 0.1
model_params["M"]             = 10
model_params["deltaS"]        = 20
model_params["max_nbr_tries"] = 10
model = Model( model_params)

nbr_samples = 1500
#epsilon     = 0.5
theta0 = max(np.array([1e-3]), problem.theta_prior_rand() )
print "INIT THETA = ",theta0
theta0 *= 0
theta0 += 0.1
state  = State( theta0, state_params )

recorder = Recorder()
recorder.record_state( state, state.nbr_sim_calls, accepted=True )

model.set_current_state( state )
model.set_recorder( recorder )
loglik = state.loglikelihood()

print "***************  RUNNING ABC MCMC ***************"
thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, model  )
print "***************  DONE ABC MCMC    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = 0 )
pp.show()
print "***************  DONE VIEW    ***************"

print "TODO"
print "verify kernel epsilon on stats"
print "improve proposal for exponential theta near 0 ... seems to get stuck"