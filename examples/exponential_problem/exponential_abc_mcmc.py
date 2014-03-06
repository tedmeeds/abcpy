from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.algos.mcmc         import abc_mcmc       
from abcpy.states.kernel_epsilon import KernelEpsilonState as State
from abcpy.states.all_states       import BaseAllStates as AllStates
from abcpy.kernels.gaussian import log_gaussian_kernel

import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = {}
problem_params["alpha"]           = 0.1
problem_params["beta"]            = 0.1
problem_params["theta_star"]      = 0.1
problem_params["N"]               = 500  # how many observations we draw per simulation
problem_params["seed"]            = 0
problem_params["q_stddev"]        = 0.2
problem = Problem( problem_params, force_init = True )


# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                          = 1
state_params["obs_statistics"]             = problem.get_obs_statistics()
state_params["theta_prior_rand_func"]      = problem.theta_prior_rand
state_params["theta_prior_logpdf_func"]    = problem.theta_prior_logpdf
state_params["theta_proposal_rand_func"]   = problem.theta_proposal_rand
state_params["theta_proposal_logpdf_func"] = problem.theta_proposal_logpdf
state_params["simulation_function"]        = problem.simulation_function
state_params["statistics_function"]        = problem.statistics_function
state_params["log_kernel_func"]            = log_gaussian_kernel
state_params["is_marginal"]                = True
state_params["epsilon"]                    = 0.2

nbr_samples = 50000
#epsilon     = 0.5
theta0 = problem.theta_prior_rand()
state  = State( theta0, state_params )
loglik = state.loglikelihood()
all_states = AllStates()
all_states.add( state, state.nbr_sim_calls, accepted=True )

print "***************  RUNNING ABC MCMC ***************"
thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, state, State, all_states  )
print "***************  DONE ABC MCMC    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( all_states, burnin = 1000 )
pp.show()
print "***************  DONE VIEW    ***************"

print "TODO"
print "verify kernel epsilon on stats"