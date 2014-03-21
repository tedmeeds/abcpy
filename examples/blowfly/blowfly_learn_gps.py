from abcpy.problems.blowfly.blowfly    import BlowflyProblem   as Problem
from abcpy.problems.blowfly.blowfly    import default_params   as load_default_params
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

import pdb
import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
problem_params["q_factor"] = 0.1
problem = Problem( problem_params, force_init = True )



model_params = {}
# adaptive-SL params
model_params["xi"]            = 20.3
model_params["M"]             = 100
model_params["deltaS"]        = 10
model_params["max_nbr_tries"] = 30

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
state_params["zero_cross_terms"]           = True # change this if we want diagonal estimates
state_params["is_marginal"]                = False
state_params["epsilon"]                    = 0.0 
state_params["hierarchy_type"]             = "just_gaussian"

print "RANDOM SEED"
np.random.seed(0)

nbr_samples = 15000
reject_epsilon     = 3.0
n_reject           = 100
nbr_thetas         = 6
nbr_stats          = 10

filename = "./examples/blowfly/gp.json"

gps = []
for gp_idx in range( nbr_stats ):
  fn = "./examples/blowfly/p%dgp.json"%((gp_idx+1))
  #json_gp = load_json( filename )
  json_gp = load_json( fn )
  #json_gp["kernel"]["type"]="squared_exponential"
  gp = build_gp_from_json( json_gp ) 
  gps.append( gp )
pgp = ProductGaussianProcess( gps) 
#assert False
surrogate_params = {}
surrogate_params["gp"] = pgp
surrogate_params["epsilon"] = 0.5
surrogate_params["obs_statistics"]     = state_params["obs_statistics"]

rej_state_params = state_params.copy()
rej_state_params["S"] = 1
rej_state = RejectState(None, rej_state_params )
recorder = Recorder(record_stats=True)

print "***************  RUNNING REJECTION ***************"
#rej_thetas, rej_discs = abc_rejection( n_reject, reject_epsilon, rej_state, recorder = recorder  )
#rej_thetas = np.loadtxt( "")
#rej_thetas = np.loadtxt("./uai2014/runs/blowfly/rejection/eps5p0_repeat1_thetas.txt")
#rej_stats  = np.loadtxt("./uai2014/runs/blowfly/rejection/eps5p0_repeat1_stats.txt")

rej_thetas = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s50/just_gaussian_repeat2_thetas.txt")[2000:,:]
perm = np.random.permutation(len(rej_thetas))
rej_thetas = rej_thetas[perm,:]
rej_stats = []
state  = State( None, state_params )
for theta in rej_thetas[:4*n_reject]:
  thetas, sim_outs, stats = state.run_sim_and_stats(theta, 1)
  rej_stats.append( np.squeeze(stats) )
  
rej_stats = np.array(rej_stats)
#rej_stats  = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s50/just_gaussian_repeat2_stats.txt")[2000:,:]
theta0 = rej_thetas[-1]
state  = State( theta0, state_params )
#assert False


surrogate_params["run_sim_and_stats_func"] = state.run_sim_and_stats
surrogate_params["update_rate"]   = 100000

surrogate = Surrogate( surrogate_params )
surrogate_params["gp"].init_with_this_data( rej_thetas[:n_reject,:].reshape( (n_reject,6)), \
                                          rej_stats[:n_reject,:].reshape( (n_reject,10)))
#surrogate_params["gp"].init_with_this_data( recorder.get_thetas().reshape( (n_reject,6)), recorder.get_statistics().reshape( (n_reject,10)))
#surrogate.update()
trainX = rej_thetas[:n_reject,:].reshape( (n_reject,6))
trainY = rej_stats[:n_reject,:].reshape( (n_reject,10))

s,v,d = np.linalg.svd( trainX )
svd_ids = [0,1,2]
projtrain = np.dot( trainX, d )
testX  = rej_thetas[n_reject:3*n_reject,:].reshape( (2*n_reject,6))
testY  = rej_stats[n_reject:3*n_reject,:].reshape( (2*n_reject,10))
projtest = np.dot( testX, d )

Ntrain = len(trainX)
Ntest  = len(testY)
idx = 0
gp = gps[idx]
#gp.optimize( method = "minimize", params = {"maxnumlinesearch":3} )  
#gp.sample( method = "slice", params = {"N":20,"MODE":2,"nbrSteps":3,"set_to_last_sample":True})
f=pp.figure()
gp = gps[idx]
sp1 = f.add_subplot( 2,2,1)
mu,cv,dcv = gp.full_posterior_mean_and_data( trainX )
var_est = np.mean( pow(mu-trainY[:,idx],2))
std_est = np.sqrt(var_est)
cv = np.sqrt(np.diag(cv))
dcv = np.sqrt(np.diag(dcv))
pp.plot( trainY[:,idx], mu, 'r.' )
trainerr = np.mean( pow( trainY[:,idx] - mu,2) )

for sp_id in range(len(svd_ids)):
  
  middle = np.argsort( projtrain[:,svd_ids[sp_id]] )[Ntrain/2]
  theta = trainX[middle,:]
  
  sp = f.add_subplot( 2,2,2+sp_id)
  pp.plot( projtrain[:,svd_ids[sp_id]], trainY[:,idx], 'r.', alpha=0.5 ) 
  pp.plot( projtrain[:,svd_ids[sp_id]], mu, 'ro', alpha=0.2 )  
  
  pp.plot( [projtrain[middle,svd_ids[sp_id]]], [mu[middle]], 'cs', alpha=1, ms=8 ) 
  pp.plot( [projtrain[middle,svd_ids[sp_id]]], [mu[middle]+std_est], 'go', alpha=1, ms=8 )  
  pp.plot( [projtrain[middle,svd_ids[sp_id]]], [mu[middle]-std_est], 'go', alpha=1, ms=8 )  
  pp.plot( [projtrain[middle,svd_ids[sp_id]]], [mu[middle]+cv[middle]], 'ys', alpha=1, ms=8 )  
  pp.plot( [projtrain[middle,svd_ids[sp_id]]], [mu[middle]-cv[middle]], 'ys', alpha=1, ms=8 )  
  pp.plot( [projtrain[middle,svd_ids[sp_id]]], [mu[middle]+dcv[middle]], 'ks', alpha=1, ms=8 )  
  pp.plot( [projtrain[middle,svd_ids[sp_id]]], [mu[middle]-dcv[middle]], 'ks', alpha=1, ms=8 ) 
  
  sp.title.set_fontsize(8)
  for tick in sp.yaxis.get_major_ticks():
    tick.label.set_fontsize(6)

  for tick in sp.xaxis.get_major_ticks():
    tick.label.set_fontsize(6)


pp.subplot(2,2,1)  
mu,cv,dcv = gp.full_posterior_mean_and_data( testX )
cv = np.sqrt(np.diag(cv))
dcv = np.sqrt(np.diag(dcv))
pp.plot( testY[:,idx], mu, 'bo', alpha=0.25 )
  
testerr = np.mean( pow( testY[:,idx] - mu,2) )
pp.title( "train %3.3f  test %3.3f"%(trainerr,testerr))
# pp.subplot(2,2,2)  
# pp.plot( projtest[:,svd_id], testY[:,idx], 'b.', alpha=0.2 ) 
# pp.plot( projtest[:,svd_id], mu, 'bo', alpha=0.2 )  

# for sp_id in range(len(svd_ids)):
#   sp = f.add_subplot( 2,2,2+sp_id)
#   pp.plot( projtest[:,svd_ids[sp_id]], testY[:,idx], 'b.', alpha=0.2 ) 
#   pp.plot( projtest[:,svd_ids[sp_id]], mu, 'bo', alpha=0.2 )  
#   
#   sp.title.set_fontsize(8)
#   for tick in sp.yaxis.get_major_ticks():
#     tick.label.set_fontsize(6)
# 
#   for tick in sp.xaxis.get_major_ticks():
#     tick.label.set_fontsize(6)
  
sp1.title.set_fontsize(8)
for tick in sp1.yaxis.get_major_ticks():
  tick.label.set_fontsize(6)

for tick in sp1.xaxis.get_major_ticks():
  tick.label.set_fontsize(6)
  # 
# sp2.title.set_fontsize(8)
# for tick in sp2.yaxis.get_major_ticks():
#   tick.label.set_fontsize(6)
# 
# for tick in sp2.xaxis.get_major_ticks():
#   tick.label.set_fontsize(6)
pp.axis("equal")



  # 
# f=pp.figure()
# idx = 0
# for gp in gps:
#   sp = f.add_subplot( 3,4,idx+1)
#   mu,cv,dcv = gp.full_posterior_mean_and_data( rej_thetas[:n_reject,:] )
#   cv = np.sqrt(np.diag(cv))
#   dcv = np.sqrt(np.diag(dcv))
#   pp.plot( rej_stats[:n_reject,idx], mu, 'r.' )
#   
#   trainerr = np.mean( pow( rej_stats[:n_reject,idx] - mu,2) )
#   #pp.plot( rej_stats[:n_reject,idx], mu+dcv, 'r.' )
#   
#   mu,cv,dcv = gp.full_posterior_mean_and_data( rej_thetas[n_reject:2*n_reject,:] )
#   cv = np.sqrt(np.diag(cv))
#   dcv = np.sqrt(np.diag(dcv))
#   pp.plot( rej_stats[n_reject:2*n_reject,idx], mu, 'bo', alpha=0.25 )
#   
#   testerr = np.mean( pow( rej_stats[n_reject:2*n_reject,idx] - mu,2) )
#   
#   pp.title( "train %3.3f  test %3.3f"%(trainerr,testerr))
#   sp.title.set_fontsize(8)
#   for tick in sp.yaxis.get_major_ticks():
#     tick.label.set_fontsize(6)
# 
#   for tick in sp.xaxis.get_major_ticks():
#     tick.label.set_fontsize(6)
#   pp.axis("equal")
#   #pp.plot( rej_stats[n_reject:2*n_reject,idx], mu+dcv, 'ro', alpha=0.5 )
#   
#   idx+=1

pp.show()