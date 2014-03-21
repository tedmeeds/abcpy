from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.problems.exponential     import default_params     as load_default_params
from abcpy.algos.model_mcmc               import abc_mcmc       
from abcpy.states.synthetic_likelihood import SyntheticLikelihoodState as State
from abcpy.states.state_recorder    import BaseStateRecorder as Recorder
from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
#from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel as MH_Model

from abcpy.states.distance_epsilon import DistanceEpsilonState as RejectState
from abcpy.algos.rejection         import abc_rejection 
import pylab as pp
import numpy as np

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
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
#state_params["log_kernel_func"]            = log_gaussian_kernel
#state_params["is_marginal"]                = True
state_params["is_marginal"]                = False
state_params["epsilon"]                    = 0.0
state_params["hierarchy_type"]             = "just_gaussian"

model_params = {}
# adaptive-SL params
model_params["M"]             = 100
model_params["max_nbr_tries"] = 30


theta0 = problem.theta_prior_rand()

#model_params["deltaS"]        = deltaS
nbr_samples  = 50000
reject_epsilon = 0.5

repeats      = 10
Ss = [2,5,10,50]
for S in Ss:
  state_params["S"]             = S

  if state_params["is_marginal"]:
    save_dir = "./uai2014/runs/exponential/sl_marginal_s%d/"%(state_params["S"])
  else:
    save_dir = "./uai2014/runs/exponential/sl_pseudo_s%d/"%(state_params["S"])
  
  for repeat in range(repeats):
    np.random.seed(repeat)
    
    rej_state_params = state_params.copy()
    rej_state_params["S"] = 1
    rej_state = RejectState(theta0, rej_state_params )

    recorder = Recorder(record_stats=False)
    n_reject = 1
    
    rej_thetas, rej_discs = abc_rejection( n_reject, reject_epsilon, rej_state, recorder = recorder  )
    theta0 = rej_thetas[-1]
    
    state  = State( theta0, state_params )
    #recorder = Recorder(record_stats=True)
    #recorder.record_state( state, state.nbr_sim_calls, accepted=False )
    
    model = MH_Model( model_params)

    model.set_current_state( state )
    model.set_recorder( recorder )
    
    #epsilon_string = "xi" + str(xi).replace(".","p")
    out_name = save_dir + state_params["hierarchy_type"]  + "_" + "repeat%d"%(repeat+1)
    
    print "***************  RUNNING ABC MCMC ***************"
    print "S = ", S
    print "is_marginal", state_params["is_marginal"] 
    print "REPEAT = ",repeat+1
    
    thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, model, verbose=False  )
    
    #print "***************  VIEW RESULTS ***************"
    #problem.view_results( recorder, burnin = 0 )
    
    print "***************  SAVING ******************************"
    recorder.save_results( out_name )
    
    assert len( recorder.get_thetas() ) == nbr_samples + n_reject
    pp.show()
    print "***************  DONE VIEW    ***************"