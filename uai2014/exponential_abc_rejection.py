from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.algos.rejection         import abc_rejection       
from abcpy.states.distance_epsilon import DistanceEpsilonState as State
from abcpy.states.state_recorder    import BaseStateRecorder as Recorder

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

theta0 = problem.theta_prior_rand()


nbr_samples  = 50000
#epsilons     = [5.0,2.0,1.0,0.5]
epsilons     = [0.1,0.05,0.01]
repeats      = 10

save_dir = "./uai2014/runs/exponential/rejection/"

for epsilon in epsilons:
  for repeat in range(repeats):
    state  = State( theta0, state_params )
    recorder = Recorder(record_stats=True)
    #recorder.record_state( state, state.nbr_sim_calls, accepted=False )
    
    epsilon_string = "eps" + str(epsilon).replace(".","p")
    out_name = save_dir + epsilon_string + "_" + "repeat%d"%(repeat+1)
    
    print "***************  RUNNING ABC REJECTION ***************"
    thetas = abc_rejection( nbr_samples, epsilon, state, recorder  )
    
    print "***************  VIEW RESULTS ***************"
    problem.view_results( recorder, burnin = 0 )
    
    print "***************  SAVING ******************************"
    recorder.save_results( out_name )
    pp.show()
    print "***************  DONE VIEW    ***************"