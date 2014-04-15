from abcpy.problems.exponential     import ExponentialProblem as Problem
from abcpy.problems.exponential     import default_params     as load_default_params
from abcpy.algos.model_mcmc               import abc_mcmc 
#from abcpy.response_kernels.epsilon_tube import EpsilonTubeResponseKernel as Kernel
from abcpy.response_kernels.epsilon_gaussian import EpsilonGaussianResponseKernel as Kernel
from abcpy.states.kernel_based_state  import KernelState as State
from abcpy.states.state_recorder      import BaseStateRecorder as Recorder
from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model

import pylab as pp
import numpy as np

problem_params = load_default_params()
problem = Problem( problem_params, force_init = True )

nbr_samples = 1500
#epsilon     = 0.5
epsilon = 0.1
kernel_params = {}
#kernel_params["lower_epsilon"]               = -np.inf
#kernel_params["upper_epsilon"]               = epsilon
kernel_params["epsilon"]                     = epsilon

state_params = {}
state_params["S"]                      = 5
state_params["observation_statistics"] = problem.get_obs_statistics()
state_params["observation_groups"]     = problem.get_obs_groups()
state_params["simulation_function"]    = problem.simulation_function
state_params["statistics_function"]    = problem.statistics_function
state_params["response_groups"]        = [Kernel( kernel_params )]

mcmc_params = {}
mcmc_params["priorrand"]         = problem.theta_prior_rand
mcmc_params["logprior"]          = problem.theta_prior_logpdf
mcmc_params["proposal_rand"]     = problem.theta_proposal_rand
mcmc_params["logproposal"]       = problem.theta_proposal_logpdf
mcmc_params["is_marginal"]       = False
mcmc_params["nbr_samples"]       = nbr_samples

model = MH_Model( mcmc_params)


theta0 = problem.theta_prior_rand()
state  = State( theta0, state_params )

recorder = Recorder(record_stats=True)

model.set_current_state( state )
model.set_recorder( recorder )
loglik = state.loglikelihood()


print "***************  RUNNING MODEL ABC MCMC ***************"
thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, model, verbose = True  )
print " ACCEPT RATE = %0.3f"%(recorder.acceptance_rate())
print "***************  DONE ABC MCMC    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = nbr_samples/2 )
pp.show()
print "***************  DONE VIEW    ***************"

print "TODO"
print "verify kernel epsilon on stats"