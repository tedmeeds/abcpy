from abcpy.problems.exponential     import ExponentialProblem   as Problem
from abcpy.problems.exponential     import default_params       as load_default_params
from abcpy.algos.rejection          import abc_rejection       
from abcpy.states.discrepancy_state  import DiscrepancyState as State
from abcpy.states.state_recorder    import BaseStateRecorder    as Recorder

import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
problem = Problem( problem_params, force_init = True )


# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                     = 1
state_params["observation_statistics"]        = problem.get_obs_statistics()
state_params["observation_groups"]     = problem.get_obs_groups()
state_params["theta_prior_rand_func"] = problem.theta_prior_rand
state_params["simulation_function"]   = problem.simulation_function
state_params["statistics_function"]   = problem.statistics_function
state_params["response_groups"]        = []
# state_params["epsilon"]               = epsilon

nbr_samples = 1000
epsilon     = 0.5
theta0 = problem.theta_prior_rand()
state  = State( theta0, state_params )
recorder = Recorder()
recorder.record_state( state, state.nbr_sim_calls, accepted=False )

print "***************  RUNNING ABC REJECTION ***************"
lower_epsilon = -epsilon
upper_epsilon = epsilon
thetas, discs = abc_rejection( nbr_samples, lower_epsilon, upper_epsilon, state, problem.theta_prior_rand, recorder  )
print "***************  DONE ABC REJECTION    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = 0 )
pp.show()
print "***************  DONE VIEW    ***************"
