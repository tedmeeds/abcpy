from abcpy.problems.exponential    import ExponentialProblem   as Problem
from abcpy.algos.model_mcmc         import abc_mcmc       
from abcpy.algos.rejection         import abc_rejection 
# from abcpy.states.kernel_epsilon import KernelEpsilonState as State
from abcpy.states.synthetic_likelihood import SyntheticLikelihoodState as State
from abcpy.states.distance_epsilon import DistanceEpsilonState as RejectState
from abcpy.states.state_recorder       import BaseStateRecorder as Recorder
from abcpy.kernels.gaussian import log_gaussian_kernel
#from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
from abcpy.metropolis_hastings_models.surrogate_metropolis_hastings_model import SurrogateMetropolisHastingsModel as MH_Model

from abcpy.surrogates.gps import GaussianProcessSurrogate as Surrogate

from progapy.gps.product_gaussian_process import ProductGaussianProcess
from progapy.factories.json2gp import load_json, build_gp_from_json
from progapy.viewers.view_1d import view as view_this_gp

import numpy as np
import pylab as pp


#np.random.seed(0)

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = {}
problem_params["alpha"]           = 0.1
problem_params["beta"]            = 0.1
problem_params["theta_star"]      = 0.1
problem_params["N"]               = 500  # how many observations we draw per simulation
problem_params["seed"]            = 0
problem_params["q_stddev"]        = 0.01
problem = Problem( problem_params, force_init = True )


# since we are running abc_rejection, use a distance epsilon state
state_params = {}
state_params["S"]                          = 2
state_params["obs_statistics"]             = problem.get_obs_statistics()
state_params["theta_prior_rand_func"]      = problem.theta_prior_rand
state_params["theta_prior_logpdf_func"]    = problem.theta_prior_logpdf
state_params["theta_proposal_rand_func"]   = problem.theta_proposal_rand
state_params["theta_proposal_logpdf_func"] = problem.theta_proposal_logpdf
state_params["simulation_function"]        = problem.simulation_function
state_params["statistics_function"]        = problem.statistics_function
state_params["log_kernel_func"]            = log_gaussian_kernel
state_params["is_marginal"]                = False
state_params["epsilon"]                    = 0.0
#state_params["hierarchy_type"]      = "jeffreys"
#state_params["hierarchy_type"]      = "jeffreys"

filename = "./examples/exponential_problem/gp.json"
json_gp = load_json( filename )
gp = build_gp_from_json( json_gp ) 
pgp = ProductGaussianProcess( [gp]) 
surrogate_params = {}
surrogate_params["gp"] = pgp
surrogate_params["obs_statistics"]     = state_params["obs_statistics"]



model_params = {}
# adaptive-SL params
model_params["xi"]            = 0.1
model_params["M"]             = 100
model_params["deltaS"]        = 5
model_params["max_nbr_tries"] = 2
model_params["gp_json"]       = json_gp

epsilon = 5.0
nbr_samples = 1500
#epsilon     = 0.5
theta0 = max(np.array([1e-3]), 0*problem.theta_prior_rand() )


print "INIT THETA = ",theta0
#theta0 *= 0
#theta0 += 0.1
state  = State( theta0, state_params )

surrogate_params["run_sim_and_stats_func"] = state.run_sim_and_stats
surrogate_params["update_rate"]   = 100

rej_state = RejectState(theta0, state_params )
thetas = abc_rejection( 20, epsilon, rej_state, recorder = None  )

surrogate = Surrogate( surrogate_params )
for theta in thetas:
  surrogate.acquire_points( theta, \
                            theta, 1, force_update=True )
gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )                            
surrogate.update()
#pgp.add_data( )

#assert False
model_params["surrogate"]     = surrogate
model = MH_Model( model_params)

recorder = Recorder()
recorder.record_state( state, state.nbr_sim_calls, accepted=True )

model.set_current_state( state )
model.set_recorder( recorder )
loglik = state.loglikelihood()

print "***************  RUNNING ABC MCMC ***************"
thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, model  )
print "***************  DONE ABC MCMC    ***************"

print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = 0 )
pp.figure()
view_this_gp( gp, x_range = [0,0.5] )
pp.show()
print "***************  DONE VIEW    ***************"

print "TODO"
print "verify kernel epsilon on stats"
print "improve proposal for exponential theta near 0 ... seems to get stuck"