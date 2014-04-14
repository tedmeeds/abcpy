from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.problems.exponential     import default_params     as load_default_params
from abcpy.algos.model_mcmc               import abc_mcmc       
#from abcpy.states.synthetic_likelihood import SyntheticLikelihoodState as State
from abcpy.states.response_model_state import ResponseModelState as State
from abcpy.response_models.gaussian_response_model import GaussianResponseModel as ResponseModel

from abcpy.states.state_recorder    import BaseStateRecorder as Recorder
#from abcpy.kernels.gaussian         import log_gaussian_kernel
from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel as MH_Model
from abcpy.acquisition_models.random_acquisition import RandomAcquisitionModel as AcquisitionModel
import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
#problem_params["q_factor"] = 0.1
problem = Problem( problem_params, force_init = True )


nbr_samples = 5000
model_params = {}
acquistion_params = {}
# adaptive-SL params
#model_params["xi"]            = 0.1
#model_params["M"]             = 10
#model_params["deltaS"]        = 5
#model_params["max_nbr_tries"] = 10


mcmc_params = {}
mcmc_params["priorrand"]         = problem.theta_prior_rand
mcmc_params["logprior"]          = problem.theta_prior_logpdf
mcmc_params["proposal_rand"]     = problem.theta_proposal_rand
mcmc_params["logproposal"]       = problem.theta_proposal_logpdf
mcmc_params["is_marginal"]       = False
mcmc_params["nbr_samples"]       = nbr_samples
mcmc_params["xi"]                = 0.1
mcmc_params["M"]                 = 10
mcmc_params["deltaS"]            = 1
mcmc_params["max_nbr_tries"]     = 10
mcmc_params["acquisition_model"] = AcquisitionModel(acquistion_params)

model = MH_Model( mcmc_params)

response_model_params = {}

# since we are running abc_rejection, use a distance epsilon state
#state_params = {}
#state_params["S"]                          = 4
#state_params["observation_statistics"]     = problem.get_obs_statistics()
#state_params["simulation_function"]        = problem.simulation_function
#state_params["statistics_function"]        = problem.statistics_function
#state_params["zero_cross_terms"]           = False # change this if we want diagonal estimates
#state_params["log_kernel_func"]            = log_gaussian_kernel
#state_params["is_marginal"]                = True
#state_params["epsilon"]                    = 0.0 #np.array([0.1,0.1,0.1,1.0])
#state_params["hierarchy_type"]             = "jeffreys"
#state_params["hierarchy_type"]             = "just_gaussian"

# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                      = 10
state_params["observation_statistics"] = problem.get_obs_statistics()
state_params["simulation_function"]    = problem.simulation_function
state_params["statistics_function"]    = problem.statistics_function
state_params["response_model"]         = ResponseModel( response_model_params )


#epsilon     = 0.5
theta0 = problem.theta_prior_rand()
#theta0 *=0
#theta0 += 0.1
state  = State( theta0, state_params )

recorder = Recorder(record_stats=True)

#recorder.record_state( state, state.nbr_sim_calls, accepted=True )

model.set_current_state( state )
model.set_recorder( recorder )
loglik = state.loglikelihood()

#recorder.record_state( state, state.nbr_sim_calls, accepted=True )

print "***************  RUNNING SL MCMC ***************"
thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, model, verbose = True  )
print " ACCEPT RATE = %0.3f"%(recorder.acceptance_rate())
print "***************  DONE ABC MCMC    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = nbr_samples/2 )
pp.show()
print "***************  DONE VIEW    ***************"

print "TODO"
print "verify kernel epsilon on stats"