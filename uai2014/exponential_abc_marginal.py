from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.algos.mcmc         import abc_mcmc       
from abcpy.algos.rejection         import abc_rejection 
# from abcpy.states.kernel_epsilon import KernelEpsilonState as State
from abcpy.states.kernel_epsilon import KernelEpsilonState as State
from abcpy.states.distance_epsilon import DistanceEpsilonState as RejectState
from abcpy.states.state_recorder       import BaseStateRecorder as Recorder
from abcpy.kernels.gaussian import log_gaussian_kernel
#from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
from abcpy.metropolis_hastings_models.surrogate_metropolis_hastings_model import SurrogateMetropolisHastingsModel as MH_Model


import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = {}
problem_params["alpha"]      = 0.1
problem_params["beta"]       = 0.1
problem_params["theta_star"] = 0.1
problem_params["N"]          = 500  # how many observations we draw per simulation
problem_params["seed"]       = 0
problem_params["q_stddev"]   = 0.01

problem = Problem( problem_params, force_init = True )


# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["obs_statistics"]             = problem.get_obs_statistics()
state_params["theta_prior_rand_func"]      = problem.theta_prior_rand
state_params["theta_prior_logpdf_func"]    = problem.theta_prior_logpdf
state_params["theta_proposal_rand_func"]   = problem.theta_proposal_rand
state_params["theta_proposal_logpdf_func"] = problem.theta_proposal_logpdf
state_params["simulation_function"]        = problem.simulation_function
state_params["statistics_function"]        = problem.statistics_function
state_params["log_kernel_func"]            = log_gaussian_kernel
state_params["is_marginal"]                = False


theta0 = problem.theta_prior_rand()

S = 10
state_params["S"]  = S
nbr_samples  = 50000
#epsilons     = [5.0,2.0,1.0]
epsilons     = [5.0,2.0,1.0,0.1,0.05,0.01]
repeats      = 10

if state_params["is_marginal"]:
  save_dir = "./uai2014/runs/exponential/abc_mcmc_marginal_s%d/"%(state_params["S"])
else:
  save_dir = "./uai2014/runs/exponential/abc_mcmc_pseudo_s%d/"%(state_params["S"])
  

for epsilon in epsilons:
  state_params["epsilon"]                    = epsilon
  for repeat in range(repeats):
    rej_state_params = state_params.copy()
    rej_state_params["S"] = 1
    rej_state = RejectState(theta0, rej_state_params )

    recorder = Recorder(record_stats=False)
    n_reject = 1
    
    rej_thetas = abc_rejection( n_reject, epsilon, rej_state, recorder = recorder  )
    theta0 = rej_thetas[-1]
    
    state  = State( theta0, state_params )
    #recorder = Recorder(record_stats=True)
    #recorder.record_state( state, state.nbr_sim_calls, accepted=False )
    
    epsilon_string = "eps" + str(epsilon).replace(".","p")
    out_name = save_dir + epsilon_string + "_" + "repeat%d"%(repeat+1)
    
    print "***************  RUNNING ABC MCMC ***************"
    thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, state, recorder  )
    
    print "***************  VIEW RESULTS ***************"
    problem.view_results( recorder, burnin = 0 )
    
    print "***************  SAVING ******************************"
    recorder.save_results( out_name )
    
    assert len( recorder.get_thetas() ) == nbr_samples + n_reject
    pp.show()
    print "***************  DONE VIEW    ***************"