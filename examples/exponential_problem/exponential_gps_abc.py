from abcpy.problems.exponential     import ExponentialProblem as Problem
from abcpy.problems.exponential     import default_params     as load_default_params
from abcpy.algos.mcmc               import abc_mcmc 
from abcpy.algos.model_mcmc               import abc_mcmc 
from abcpy.response_kernels.epsilon_tube import EpsilonTubeResponseKernel as Kernel
from abcpy.response_kernels.epsilon_gaussian import EpsilonGaussianResponseKernel as Kernel
from abcpy.states.kernel_based_state  import KernelState as State
from abcpy.states.state_recorder      import BaseStateRecorder as Recorder
from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel as MH_Model
from abcpy.states.response_model_state import ResponseModelState as State
from abcpy.response_models.surrogate_response_model import SurrogateResponseModel as ResponseModel
#from abcpy.response_models.gaussian_response_model import GaussianResponseModel as ResponseModel
from abcpy.acquisition_models.random_acquisition import RandomAcquisitionModel as AcquisitionModel

from abcpy.surrogates.gps import GaussianProcessSurrogate as Surrogate
from progapy.gps.product_gaussian_process import ProductGaussianProcess
from progapy.factories.json2gp import load_json, build_gp_from_json
from progapy.viewers.view_1d import view as view_this_gp


import pylab as pp
import numpy as np

problem_params = load_default_params()
problem = Problem( problem_params, force_init = True )

nbr_samples = 5000
epsilon = 0.1

filename = "./examples/exponential_problem/gp.json"
json_gp = load_json( filename )
gp = build_gp_from_json( json_gp ) 
pgp = ProductGaussianProcess( [gp] ) 
surrogate_params = {}
surrogate_params["gp"] = pgp
surrogate = Surrogate( surrogate_params )
response_model_params = {"surrogate":surrogate}
acquistion_params = {}

state_params = {}
state_params["S"]                      = 5
state_params["observation_statistics"] = problem.get_obs_statistics()
state_params["observation_groups"]     = problem.get_obs_groups()
state_params["simulation_function"]    = problem.simulation_function
state_params["statistics_function"]    = problem.statistics_function
state_params["response_groups"]        = [ResponseModel( response_model_params )]

mcmc_params = {}
mcmc_params["priorrand"]         = problem.theta_prior_rand
mcmc_params["logprior"]          = problem.theta_prior_logpdf
mcmc_params["proposal_rand"]     = problem.theta_proposal_rand
mcmc_params["logproposal"]       = problem.theta_proposal_logpdf
mcmc_params["is_marginal"]       = False
mcmc_params["nbr_samples"]       = nbr_samples
mcmc_params["xi"]                = 0.4
mcmc_params["M"]                 = 10
mcmc_params["deltaS"]            = 1
mcmc_params["max_nbr_tries"]     = 1
mcmc_params["acquisition_model"] = AcquisitionModel(acquistion_params)

model = MH_Model( mcmc_params)


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