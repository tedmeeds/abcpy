from abcpy.problems.optimization.three_bumps     import ThreeBumpsProblem   as Problem
from abcpy.problems.optimization.three_bumps     import default_params       as load_default_params
from abcpy.algos.mcmc               import abc_mcmc   
from abcpy.algos.model_mcmc               import abc_mcmc 
from abcpy.response_kernels.epsilon_tube import EpsilonTubeResponseKernel as Kernel
from abcpy.response_kernels.epsilon_gaussian import EpsilonGaussianResponseKernel as Kernel
from abcpy.response_kernels.epsilon_heavyside_gaussian import EpsilonHeavysideGaussianResponseKernel as Kernel
from abcpy.states.kernel_based_state    import KernelState as State
#from abcpy.states.distance_epsilon  import DistanceEpsilonState as State
from abcpy.states.state_recorder    import BaseStateRecorder    as Recorder
from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
#from abcpy.kernels.gaussian         import one_sided_gaussian_kernel as kernel
#from abcpy.kernels.gaussian         import one_sided_exponential_kernel as kernel
from abcpy.helpers import logsumexp
import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
problem_params["noise"] = 0.0001
problem_params["prior_mu"] = 0
problem_params["prior_std"] = 2
problem_params["q_stddev"] = 0.75
problem = Problem( problem_params, force_init = True )

epsilon     = 0.05
kernel_params = {}
kernel_params["lower_epsilon"]               = np.array([-np.inf])
kernel_params["upper_epsilon"]               = np.array([epsilon])
kernel_params["epsilon"]                     = np.array([epsilon])
kernel_params["direction"]                   = "down"

# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                      = 2
state_params["observation_statistics"] = problem.get_obs_statistics()
state_params["observation_groups"]     = problem.get_obs_groups()
state_params["simulation_function"]    = problem.simulation_function
state_params["statistics_function"]    = problem.statistics_function
state_params["response_groups"]        = [Kernel( kernel_params )]
#state_params["kernel"]                 = Kernel( kernel_params )


mcmc_params = {}
mcmc_params["priorrand"]       = problem.theta_prior_rand
mcmc_params["logprior"]        = problem.theta_prior_logpdf
mcmc_params["proposal_rand"]   = problem.theta_proposal_rand
mcmc_params["logproposal"]     = problem.theta_proposal_logpdf
mcmc_params["is_marginal"]     = True

nbr_samples = 10000
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
problem.view_results( recorder, burnin = 0, epsilon = epsilon )

# #d=problem.ystar+epsilon-problem.simulation_mean_function(problem.fine_theta_range)
# #I=find(d>=0)
# #pdb.set_trace()
# loglikelihood = np.squeeze(state.response_groups[0].loglikelihood(problem.ystar, problem.simulation_mean_function(problem.fine_theta_range)))
# logposterior = loglikelihood + problem.theta_prior_logpdf(problem.fine_theta_range)
# #logposterior -= logsumexp(logposterior)
# posterior = np.exp(logposterior)
# Z = np.sum(0.5*(posterior[1:]+posterior[:-1])*problem.fine_bin_width)
# #Z=2*posterior.sum()*problem.fine_bin_width
# pp.plot( problem.fine_theta_range, posterior/Z, "b",lw=2)
pp.show()
print "***************  DONE VIEW    ***************"
