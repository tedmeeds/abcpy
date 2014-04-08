from abcpy.problems.optimization.three_bumps     import ThreeBumpsProblem   as Problem
from abcpy.problems.optimization.three_bumps     import default_params       as load_default_params
from abcpy.algos.mcmc               import abc_mcmc       
from abcpy.states.kernel_epsilon    import KernelEpsilonState as State
#from abcpy.states.distance_epsilon  import DistanceEpsilonState as State
from abcpy.states.state_recorder    import BaseStateRecorder    as Recorder
from abcpy.kernels.gaussian         import one_sided_gaussian_kernel as kernel
from abcpy.kernels.gaussian         import one_sided_exponential_kernel as kernel
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

epsilon     = 0.1

# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                     = 1
state_params["obs_statistics"]        = problem.get_obs_statistics()
state_params["theta_prior_rand_func"] = problem.theta_prior_rand
state_params["theta_prior_logpdf_func"] = problem.theta_prior_logpdf
state_params["theta_proposal_rand_func"] = problem.theta_proposal_rand
state_params["theta_proposal_logpdf_func"] = problem.theta_proposal_logpdf
state_params["simulation_function"]   = problem.simulation_function
state_params["statistics_function"]   = problem.statistics_function
state_params["log_kernel_func"]            = kernel
state_params["is_marginal"]                = False
state_params["epsilon"]               = epsilon

nbr_samples = 50000
theta0 = problem.theta_prior_rand()
state  = State( theta0, state_params )
recorder = Recorder()
recorder.record_state( state, state.nbr_sim_calls, accepted=False )

print "***************  RUNNING ABC REJECTION ***************"
thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, state, recorder  )
print "***************  DONE ABC REJECTION    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = 0, epsilon = epsilon )

#d=problem.ystar+epsilon-problem.simulation_mean_function(problem.fine_theta_range)
#I=find(d>=0)
loglikelihood = np.squeeze(kernel(problem.simulation_mean_function(problem.fine_theta_range),problem.ystar,epsilon))
logposterior = loglikelihood + problem.theta_prior_logpdf(problem.fine_theta_range)
#logposterior -= logsumexp(logposterior)
posterior = np.exp(logposterior)
Z = np.sum(0.5*(posterior[1:]+posterior[:-1])*problem.fine_bin_width)
#Z=2*posterior.sum()*problem.fine_bin_width
pp.plot( problem.fine_theta_range, posterior/Z, "b",lw=2)
pp.show()
print "***************  DONE VIEW    ***************"
