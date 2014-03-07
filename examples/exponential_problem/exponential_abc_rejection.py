from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.algos.rejection         import abc_rejection       
from abcpy.states.distance_epsilon import DistanceEpsilonState as State
from abcpy.states.all_states       import BaseAllStates as AllStates

import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = {}
problem_params["alpha"]      = 0.1
problem_params["beta"]       = 0.1
problem_params["theta_star"] = 0.1
problem_params["N"]          = 500  # how many observations we draw per simulation
problem_params["seed"]       = 0

problem = Problem( problem_params, force_init = True )


# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                     = 1
state_params["obs_statistics"]        = problem.get_obs_statistics()
state_params["theta_prior_rand_func"] = problem.theta_prior_rand
state_params["simulation_function"]   = problem.simulation_function
state_params["statistics_function"]   = problem.statistics_function
# state_params["epsilon"]               = epsilon

nbr_samples = 1000
epsilon     = 2.5
theta0 = problem.theta_prior_rand()
state  = State( theta0, state_params )
all_states = AllStates()
all_states.add( state, state.nbr_sim_calls, accepted=False )

print "***************  RUNNING ABC REJECTION ***************"
thetas = abc_rejection( nbr_samples, epsilon, state, State, all_states  )
print "***************  DONE ABC REJECTION    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( all_states, burnin = 0 )
pp.show()
print "***************  DONE VIEW    ***************"
