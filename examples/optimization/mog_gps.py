from abcpy.problems.mog.mog_problem     import MogProblem   as Problem
from abcpy.problems.mog.mog_problem     import default_params       as load_default_params
from abcpy.algos.mcmc               import abc_mcmc 
from abcpy.algos.model_mcmc               import abc_mcmc 
from abcpy.response_kernels.epsilon_tube import EpsilonTubeResponseKernel as Kernel
from abcpy.response_kernels.epsilon_gaussian import EpsilonGaussianResponseKernel as Kernel
from abcpy.states.kernel_based_state  import KernelState as State
from abcpy.states.state_recorder      import BaseStateRecorder as Recorder
from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel as MH_Model
from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel as MH_Model
from abcpy.states.response_model_state import ResponseModelState as State
from abcpy.response_models.surrogate_response_model import SurrogateResponseModel as ResponseModel
#from abcpy.response_models.gaussian_response_model import GaussianResponseModel as ResponseModel
from abcpy.acquisition_models.random_acquisition import RandomAcquisitionModel as AcquisitionModel
#from abcpy.acquisition_models.mhe_acquisition import MetropolisHastingsErrorAcquisitionModel as AcquisitionModel

from abcpy.surrogates.gps import GaussianProcessSurrogate as Surrogate
from progapy.gps.product_gaussian_process import ProductGaussianProcess
from progapy.factories.json2gp import load_json, build_gp_from_json
from progapy.viewers.view_1d import view as view_this_gp
from abcpy.helpers import logsumexp
import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
problem_params["noise"] = 1.5
p1 = 0.05
problem_params["optimization"] = True
problem_params["bump_weights"]  = np.array([p1,1-p1])
# problem_params["prior_mu"] = 0
# problem_params["prior_std"] = 1
problem_params["q_stddev"] = 2.0
problem = Problem( problem_params, force_init = True )

print "================================================"
print "================================================"
print "setting random seed !!"
print "================================================"
print "================================================"
np.random.seed(1)
epsilon     = 0.25

nbr_samples = 500

response_model_params = {"likelihood_type":"logcdfcomplement"}
filename = "./examples/optimization/mog_gp.json"
# json_gp = load_json( filename )
# gp = build_gp_from_json( json_gp ) 
# pgp = ProductGaussianProcess( [gp] ) 
# surrogate_params = {}
# surrogate_params["gp"] = pgp
# surrogate = Surrogate( surrogate_params )
#response_model_params = {"surrogate":surrogate}
acquistion_params = {"nbr_steps":2}
  
state_params = {}
state_params["S"]                      = 5
state_params["observation_statistics"] = problem.get_obs_statistics()
state_params["observation_groups"]     = problem.get_obs_groups()
state_params["simulation_function"]    = problem.simulation_function
state_params["statistics_function"]    = problem.statistics_function
state_params["length_scale_update"]    = True

json_gp = load_json( filename )
J = 1
response_groups=[]
for og in state_params["observation_groups"] :
  surrogate_params = {}
  surrogate_params["gp"] = build_gp_from_json( json_gp )
  surrogate = Surrogate( surrogate_params )
  surrogate.gp.noise.set_params( np.array( [problem_params["noise"]] ))
  response_model_params = {"surrogate":surrogate,"likelihood_type":og.params["likelihood_type"]}
  response_groups.append( ResponseModel( response_model_params ) )
  
  initX = []
  initY = []
  ninit =10
  for i in range(ninit):
    initX.append( problem.theta_prior_rand() )
    initY.append( problem.simulation_function( np.array([initX[-1]])))
  initX = np.array(initX).reshape( (ninit,1))
  initY = np.array(initY).reshape( (ninit,1))
  
  surrogate.gp.add_data( initX, initY )
  surrogate.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
  surrogate.update_post_mh( og, None, None )
  print surrogate_params["gp"].N
  
state_params["response_groups"]        = response_groups #[ResponseModel( response_model_params )]

mcmc_params = {}
mcmc_params["priorrand"]         = problem.theta_prior_rand
mcmc_params["logprior"]          = problem.theta_prior_logpdf
mcmc_params["proposal_rand"]     = problem.theta_proposal_rand
mcmc_params["logproposal"]       = problem.theta_proposal_logpdf
mcmc_params["is_marginal"]       = False
mcmc_params["nbr_samples"]       = nbr_samples
mcmc_params["xi"]                = 0.1
mcmc_params["M"]                 = 100
mcmc_params["deltaS"]            = 1
mcmc_params["max_nbr_tries"]     = 3
mcmc_params["acquisition_model"] = AcquisitionModel(acquistion_params)

model = MH_Model( mcmc_params)


#epsilon     = 0.5
theta0 = problem.theta_prior_rand()

state  = State( theta0, state_params )

recorder = Recorder(record_stats=True)

#recorder.record_state( state, state.nbr_sim_calls, accepted=True )

model.set_current_state( state )
model.set_recorder( recorder )
#loglik = state.loglikelihood()

#recorder.record_state( state, state.nbr_sim_calls, accepted=True )

print "***************  RUNNING SL MCMC ***************"
thetas, LL, acceptances,sim_calls = abc_mcmc( nbr_samples, model, verbose = True  )

# for rg in response_groups:
#   #rg.surrogate.gp.train() #add_data( initX, initY )
#   rg.surrogate.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
print " ACCEPT RATE = %0.3f"%(recorder.acceptance_rate())
print "***************  DONE ABC MCMC    ***************"
pp.figure()
view_this_gp( response_groups[0].surrogate.gp, (-4,4) )
ax = pp.axis()
x_space = np.linspace( ax[0], ax[1], 200 )
y_est = problem.simulation_mean_function( x_space )
pp.plot( x_space, y_est, 'r-', lw=2)

problem.ystar = state_params["observation_groups"][0].ystar[0]
print "***************  VIEW RESULTS ***************"
problem.view_results( recorder, burnin = nbr_samples/2 )
pp.show()
print "***************  DONE VIEW    ***************"

print "TODO"
print "verify kernel epsilon on stats"