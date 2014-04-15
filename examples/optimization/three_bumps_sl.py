from abcpy.problems.optimization.three_bumps     import ThreeBumpsProblem   as Problem
from abcpy.problems.optimization.three_bumps     import default_params       as load_default_params
from abcpy.algos.model_mcmc               import abc_mcmc       
from abcpy.states.response_model_state import ResponseModelState as State
from abcpy.response_models.gaussian_response_model import GaussianResponseModel as ResponseModel

from abcpy.states.state_recorder    import BaseStateRecorder as Recorder
from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
#from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel as MH_Model
from abcpy.acquisition_models.random_acquisition import RandomAcquisitionModel as AcquisitionModel
from abcpy.helpers import logsumexp
import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
problem_params["noise"] = 0.1
problem_params["prior_mu"] = 0
problem_params["prior_std"] = 1
problem_params["q_stddev"] = 0.75
problem = Problem( problem_params, force_init = True )

response_model_params = {"diagonalize":True, "likelihood_type":"logcdf"}
acquistion_params = {}
epsilon     = 0.25

nbr_samples = 5000
mcmc_params = {}
mcmc_params["priorrand"]         = problem.theta_prior_rand
mcmc_params["logprior"]          = problem.theta_prior_logpdf
mcmc_params["proposal_rand"]     = problem.theta_proposal_rand
mcmc_params["logproposal"]       = problem.theta_proposal_logpdf
mcmc_params["is_marginal"]       = True
mcmc_params["nbr_samples"]       = nbr_samples
mcmc_params["xi"]                = 0.1
mcmc_params["M"]                 = 100
mcmc_params["deltaS"]            = 5
mcmc_params["max_nbr_tries"]     = 10
mcmc_params["acquisition_model"] = AcquisitionModel(acquistion_params)

model = MH_Model( mcmc_params)



# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                      = 10
state_params["observation_statistics"] = problem.get_obs_statistics()
state_params["simulation_function"]    = problem.simulation_function
state_params["statistics_function"]    = problem.statistics_function
state_params["response_model"]         = ResponseModel( response_model_params )

#epsilon     = 0.5
theta0 = problem.theta_prior_rand()
theta0 *=0
theta0 += 0.1
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