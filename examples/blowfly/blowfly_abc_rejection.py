from abcpy.factories import *
from abcpy.problems.blowfly.blowfly import *
import pylab as pp

problem_params = default_params()
problem_params["epsilon"] = 2
problem = BlowflyProblem( problem_params, force_init = True )

# since we are running abc_rejection, use a distance epsilon state
state_params = state_params_factory.scrape_params_from_problem( problem )

algo_params = { "modeling_approach"  : "kernel",
                "observation_groups" : problem.get_obs_groups(),
                "state_params"       : state_params,
                "algorithm"          : "rejection"
              }
recorder_params = {}  
algo, state  = algo_factory.create_algo_and_state( algo_params )
recorder     = recorder_factory.create_recorder( recorder_params )

nbr_samples = 100
print "***************  RUNNING ABC REJECTION ***************"
thetas, discs = algo( nbr_samples, \
                      -problem_params["epsilon"], \
                      problem_params["epsilon"], \
                      state, \
                      problem.theta_prior_rand, \
                      recorder  )
print "***************  DONE ABC REJECTION    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = 0 )
print "***************  DONE VIEW    ***************"

pp.show()