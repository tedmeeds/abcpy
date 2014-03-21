from abcpy.problems.blowfly.blowfly    import BlowflyProblem   as Problem
from abcpy.problems.blowfly.blowfly    import default_params   as load_default_params
from abcpy.algos.rejection         import abc_rejection       
from abcpy.states.distance_epsilon import DistanceEpsilonState as State
from abcpy.states.state_recorder    import BaseStateRecorder as Recorder

import pylab as pp

problem_params = load_default_params()
problem = Problem( problem_params, force_init = True )


# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                     = 1
state_params["obs_statistics"]        = problem.get_obs_statistics()
state_params["theta_prior_rand_func"] = problem.theta_prior_rand
state_params["simulation_function"]   = problem.simulation_function
state_params["statistics_function"]   = problem.statistics_function
# state_params["epsilon"]               = epsilon

nbr_samples = 100
epsilon     = 15.0
theta0 = problem.theta_prior_rand()
state  = State( theta0, state_params )
recorder = Recorder(record_stats=True)

print "***************  RUNNING ABC REJECTION ***************"
thetas, discs = abc_rejection( nbr_samples, epsilon, state, recorder  )
print "***************  DONE ABC REJECTION    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = 0 )
pp.show()
print "***************  DONE VIEW    ***************"
