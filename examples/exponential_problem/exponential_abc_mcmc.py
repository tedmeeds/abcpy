from abcpy.problems.exponential     import ExponentialProblem as Problem
from abcpy.problems.exponential     import default_params     as load_default_params
from abcpy.algos.mcmc               import abc_mcmc       
from abcpy.response_kernels.epsilon_tube import EpsilonTubeResponseKernel as Kernel
#from abcpy.response_kernels.epsilon_gaussian import EpsilonGaussianResponseKernel as Kernel
from abcpy.states.kernel_based_state  import KernelState as State
from abcpy.states.state_recorder      import BaseStateRecorder as Recorder

import pylab as pp
import numpy as np

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
problem = Problem( problem_params, force_init = True )

nbr_samples = 1500
#epsilon     = 0.5
epsilon = 0.1
kernel_params = {}
#kernel_params["lower_epsilon"]               = -np.inf
#kernel_params["upper_epsilon"]               = epsilon
kernel_params["epsilon"]                     = epsilon
#kernel_params["direction"]                   = "down"

# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                      = 5
state_params["observation_statistics"] = problem.get_obs_statistics()
state_params["simulation_function"]    = problem.simulation_function
state_params["statistics_function"]    = problem.statistics_function
state_params["kernel"]                 = Kernel( kernel_params )


mcmc_params = {}
mcmc_params["priorrand"]       = problem.theta_prior_rand
mcmc_params["logprior"]        = problem.theta_prior_logpdf
mcmc_params["proposal_rand"]   = problem.theta_proposal_rand
mcmc_params["logproposal"]     = problem.theta_proposal_logpdf
mcmc_params["is_marginal"]     = False
mcmc_params["nbr_samples"] = nbr_samples

theta0 = problem.theta_prior_rand()
state  = State( 0.1+0*theta0, state_params )
loglik = state.loglikelihood()
recorder = Recorder()
recorder.record_state( state, state.nbr_sim_calls, accepted=True )

print "***************  RUNNING ABC MCMC ***************"
thetas, LL, acceptances,sim_calls = abc_mcmc( mcmc_params, state, recorder  )
print "***************  DONE ABC MCMC    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = nbr_samples/2 )
pp.show()
print "***************  DONE VIEW    ***************"

print "TODO"
print "verify kernel epsilon on stats"