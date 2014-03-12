from abcpy.problem import BaseProblem
from abcpy.plotting import *
from abcpy.helpers import *
import numpy as np
import scipy as sp
import pylab as pp

import pdb

def default_params():
  mu_log_P         = 2.0
  std_log_P        = 2.0
  mu_log_delta     = -1.0
  std_log_delta    = 2.0
  mu_log_N0        = 5.0
  std_log_N0       = 2.0
  mu_log_sigma_d   = 0.0
  std_log_sigma_d  = 2.0
  mu_log_sigma_p   = 0.0
  std_log_sigma_p  = 2.0
  mu_tau           = 15
  q_factor         = 0.01

  params = {}
  params["blowfly_filename"] = "./problems/blowfly/blowfly.txt"
  params["mu_log_P"]         = mu_log_P
  params["std_log_P"]        = std_log_P
  params["mu_log_delta"]     = mu_log_delta
  params["std_log_delta"]    = std_log_delta
  params["mu_log_N0"]        = mu_log_N0     
  params["std_log_N0"]       = std_log_N0
  params["mu_log_sigma_d"]   = mu_log_sigma_d
  params["std_log_sigma_d"]  = std_log_sigma_d
  params["mu_log_sigma_p"]   = mu_log_sigma_p
  params["std_log_sigma_p"]  = std_log_sigma_p
  params["mu_tau"]           = mu_tau
  params["q_factor"]         = q_factor
  
  return params
  
class BlowflyProblem( BaseProblem ):
  # extract info about specific for this problem
  def load_params( self, params ):
    # which blowfly data are we using
    self.blowfly_filename = params["blowfly_filename"]
    self.theta_names = ["log_P","log_delta","log_N0","log_sigma_d","log_sigma_p","tau"]
    self.stats_names = ["mean","s0-median","peaks","log max"]
    # each parameter except for tau is in log-space, and it has a gaussian prior
    self.mu_log_P         = params["mu_log_P"]
    self.std_log_P        = params["std_log_P"]
    self.mu_log_delta     = params["mu_log_delta"]
    self.std_log_delta    = params["std_log_delta"]
    self.mu_log_N0        = params["mu_log_N0"]        
    self.std_log_N0       = params["std_log_N0"]
    self.mu_log_sigma_d   = params["mu_log_sigma_d"]
    self.std_log_sigma_d  = params["std_log_sigma_d"]
    self.mu_log_sigma_p   = params["mu_log_sigma_p"]   
    self.std_log_sigma_p  = params["std_log_sigma_p"]
    self.mu_tau           = params["mu_tau"]
    
    # a factor of the prior's stddev for the proposal stdev
    self.q_factor = params["q_factor"]
    
    
  # "create" problem or load observations  
  def initialize( self ):
    assert self.initialized is False, "Ensure we only call this once..."
    
    # load observations and generate its statistics
    self.observations   = np.loadtxt( self.blowfly_filename )[:,1] # last column has values
    self.obs_statistics = self.statistics_function( self.observations )
    
    self.T = len(self.observations)
    self.nbr_parameters = 6
    
    # done initialization
    self.initialized = True
    
  def get_observations( self ):
    assert self.initialized, "Not initialized..."
    return self.observations
    
  def get_obs_statistics( self ):
    assert self.initialized, "Not initialized..."
    return self.obs_statistics
      
  # run simulation at parameter setting theta, return outputs
  def simulation_function( self, theta ):
    # NB: this is equation (1) in supplementaty information of Wood (2010) ("A better alternative model")
    log_P       = theta[0]
    log_delta   = theta[1]
    log_N0      = theta[2]
    log_sigma_d = theta[3]
    log_sigma_p = theta[4]
    tau         = theta[5]
    
    N0, P, tau, sigma_p, sigma_d, delta = np.exp(log_N0), np.exp(log_P), tau, np.exp(log_sigma_p), np.exp(log_sigma_d), np.exp(log_delta)
  
    T = self.T
  
    var_d  = sigma_d**2
    prec_d = 1.0 / var_d

    var_p  = sigma_p**2
    prec_p = 1.0 / var_p

    burnin = 50
    lag = int(np.floor(tau))
    if (float(tau)-float(lag)>0.5):
      lag = lag + 1

    N = np.zeros( lag+burnin+T, dtype=float)
    #print N0
    N[0] = N0

    for i in range(lag):
      N[i] = 180.0

    for i in xrange(burnin+T):
      t = i + lag

      eps_t = gamma_rnd( prec_d, prec_d )
      e_t   = gamma_rnd( prec_p, prec_p )

      #tau_t = max(0,t-int(tau))
      tau_t = t - lag
      N[t] = P*N[tau_t]*np.exp(-N[tau_t]/N0)*e_t + N[t-1]*np.exp(-delta*eps_t)
  
    return N[-(T+1):]

  # pass outputs through statistics function, return statistics
  def statistics_function( self, outputs ):
    nstats = 4
    s = np.zeros( nstats, dtype = float )
    s[0] = outputs.mean() / 1000.0
    s[1] = (s[0] - np.median(outputs))/ 1000.0
    mx,mn = peakdet(outputs/outputs.std(), 1.5 )
    s[2] = float(len(mx))
    s[3] = np.log(np.max(outputs+1)/1000.0)
    return s
    # def blowstats( N ):
    #       maxlags = 11
    #   
    #       lags = 2*np.array( [6,6,6,1,1] )
    #       pows = np.array( [1,2,3,1,2] )
    #       n_reg_coeffs = len(lags)
    #   
    #       nstats = 2 + maxlags + n_reg_coeffs + 1
    #       #nstats -= maxlags
    #       nstats = 4
    #       s = np.zeros( nstats, dtype = float )
    #   
    #       s[0] = N.mean() / 1000.0
    #       s[1] = (s[0] - np.median(N))/ 1000.0
    #   
    #       mx,mn = peakdet(N/N.std(), 1.5 )
    #       #mx,mn = peakdet( N, 20000.0 )
    #       #pdb.set_trace()
    #       s[2] = float(len(mx))
    #       #lag,ac,dum1,dum2 = pp.acorr( N/1000.0, normed=True, maxlags=maxlags)
    #   
    #       #s[2:2+maxlags] = ac[:maxlags]
    #   
    #       #reg_coeffs = compute_regression( N/1000.0, lags, pows )
    #   
    #       #d1 = np.diff( N/1000.0 )
    #       #d2 = np.diff( d1 )
    #   
    #       #s[2] = np.mean(d1)
    #       #s[3] = np.mean(d2)
    #       #pdb.set_trace()
    #       #s[2:2+5] = reg_coeffs
    #       #s[2+maxlags:-1] = reg_coeffs
    #   
    #       #s[-2] = float( len(pp.find( np.abs(np.diff(np.sign(np.diff(N))))>0)) )
    #       s[-1] = np.log(np.max(N+1)/1000.0)
    #       #s[-2] = np.log(np.min(N+1)/1000.0)
    #       #s[-2] = np.max(np.abs( N[5:]-N[:-5] ))/1000.0
    #       #print "ADD turning points"
    #   
    #       return s
    return np.array( [np.mean( outputs )] )
    
  # return size of statistics vector for this problem
  def get_nbr_statistics( self ):
    return len(self.obs_statistics)
  
  # theta_rand
  def theta_prior_rand( self, N=1 ):
    theta = np.zeros((N,self.nbr_parameters))
    
    theta[:,0] = self.mu_log_P       + self.std_log_P*np.random.randn( N ) # np.log(P)
    theta[:,1] = self.mu_log_delta   + self.std_log_delta*np.random.randn( N )# np.log(delta)
    theta[:,2] = self.mu_log_N0      + self.std_log_N0*np.random.randn( N ) # np.log(N0)
    theta[:,3] = self.mu_log_sigma_d + self.std_log_sigma_d*np.random.randn( N )# np.log(sigma_d)
    theta[:,4] = self.mu_log_sigma_p + self.std_log_sigma_p*np.random.randn( N ) # np.log(sigma_p)
    theta[:,5] = poisson_rand( self.mu_tau ) # tau
    
    return np.squeeze(theta)
    
  def theta_prior_logpdf( self, theta ):
    log_p = 0.0
    
    log_p += gaussian_logpdf( theta[0], self.mu_log_P,       self.std_log_P )
    log_p += gaussian_logpdf( theta[1], self.mu_log_delta,   self.std_log_delta )
    log_p += gaussian_logpdf( theta[2], self.mu_log_N0,      self.std_log_N0 )
    log_p += gaussian_logpdf( theta[3], self.mu_log_sigma_d, self.std_log_sigma_d )
    log_p += gaussian_logpdf( theta[4], self.mu_log_sigma_p, self.std_log_sigma_p )
    log_p += poisson_logpdf(  theta[5], self.mu_tau )
    
    return log_p
      
  def theta_proposal_rand( self, theta ):
    
    tau = theta[5]
    u = np.random.rand()
    if u < 0.25 and tau > 1:
      delta_tau = -1
    elif u >=0.75:
      delta_tau = 1
    else:
      delta_tau = 0
      
    q_theta = np.zeros(self.nbr_parameters)
    
    q_theta[0] = theta[0] + self.q_factor*self.std_log_P*np.random.randn(  ) # np.log(P)
    q_theta[1] = theta[1] + self.q_factor*self.std_log_delta*np.random.randn(  )# np.log(delta)
    q_theta[2] = theta[2] + self.q_factor*self.std_log_N0*np.random.randn(  ) # np.log(N0)
    q_theta[3] = theta[3] + self.q_factor*self.std_log_sigma_d*np.random.randn(  )# np.log(sigma_d)
    q_theta[4] = theta[4] + self.q_factor*self.std_log_sigma_p*np.random.randn(  ) # np.log(sigma_p)
    q_theta[5] = theta[5] + delta_tau  # tau
    
    return q_theta
    
  def theta_proposal_logpdf( self, to_theta, from_theta ):
    log_p = 0.0
    
    log_p += gaussian_log_pdf( to_theta[0], from_theta[0], self.q_factor*self.std_log_P )
    log_p += gaussian_log_pdf( to_theta[1], from_theta[1], self.q_factor*self.std_log_delta )
    log_p += gaussian_log_pdf( to_theta[2], from_theta[2], self.q_factor*self.std_log_N0 )
    log_p += gaussian_log_pdf( to_theta[3], from_theta[3], self.q_factor*self.std_log_sigma_d )
    log_p += gaussian_log_pdf( to_theta[4], from_theta[4], self.q_factor*self.std_log_sigma_p )
    
    delta_tau = np.abs(from_tau - to_tau)
    if delta_tau > 0:
      log_p += np.log(0.25)
    else:
      log_p += np.log(0.5)
      
    log_p += poisson_log_pdf(  theta[5], self.mu_tau )
    return log_p
      
  # take samples/staistics etc and "view" this particular problem
  def view_results( self, states_object, burnin = 1 ):
    # plotting params
    nbins       = 20
    alpha       = 0.5
    label_size  = 8
    linewidth   = 3
    linecolor   = "r"
    
    # extract from states
    thetas = states_object.get_thetas(burnin=burnin)
    stats  = states_object.get_statistics(burnin=burnin)
    nsims  = states_object.get_sim_calls(burnin=burnin)
    
    f=pp.figure()
    for i in range(6):
      f.add_subplot(2,6,i+1)
      pp.hist( thetas[:,i], 10, normed=True, alpha = 0.5)
      pp.title( self.theta_names[i])
    for i in range(4):
      f.add_subplot(2,6,6+1+i+1)
      pp.hist( stats[:,i], 10, normed=True, alpha = 0.5)
      ax=pp.axis()
      pp.vlines( self.obs_statistics[i], 0, ax[3], color="r", linewidths=2)
      pp.axis(ax)
      pp.title( self.stats_names[i])
      
if __name__ == "__main__":
  pp.close("all")
  N0      = 450.0 #3721.0 #np.exp(6.0)
  sigma_p = 1.1 #np.exp(-0.5)
  sigma_d = 0.1 #np.exp(-0.75) # smoothness
  tau     = 15.0
  P       = 12.0 #np.exp(2.0)
  delta   = 0.9# np.exp(-1.8)
  Nr      = 5
  
  # N0      = 450.0 #np.exp(6.0)
 #  sigma_p = 1.5#np.exp(-0.5)
 #  sigma_d = 0.5 #np.exp(-0.75) # smoothness
 #  tau     = 15.0
 #  P       = 2.25 #np.exp(2.0)
 #  delta   = 0.24# np.exp(-1.8)
  
  params = default_params()
  
  theta = np.zeros(6)
  theta[0] = np.log(P)
  theta[1] = np.log(delta)
  theta[2] = np.log(N0)
  theta[3] = np.log(sigma_d)
  theta[4] = np.log(sigma_p)
  theta[5] = tau
  
  theta_test = theta
  b = BlowflyProblem( params, force_init = True )
  test_obs = b.simulation_function( theta_test )
  pp.figure(1)
  pp.clf()
  pp.plot( b.observations/1000.0 )
  pp.plot( test_obs / 1000.0)
  
  pp.figure(2)
  min_err = np.inf
  min_theta=None
  for i in range(16):
    pp.subplot(4,4,i+1)
    theta = b.theta_prior_rand()
    test_obs = b.simulation_function( theta )
    test_stats = b.statistics_function( test_obs )
    err = np.sum( np.abs( b.obs_statistics - test_stats ) )
    if err < min_err:
      min_err = err
      min_theta = theta.copy()
    pp.title( "%0.2f"%( err ))
    pp.plot( b.observations/1000.0 )
    pp.plot(test_obs/1000.0)
    pp.axis("off")
  pp.figure(3)
  theta = min_theta
  for i in range(16):
    pp.subplot(4,4,i+1)
    
    test_obs = b.simulation_function( theta )
    test_stats = b.statistics_function( test_obs )
    err = np.sum( np.abs( b.obs_statistics - test_stats ) )
    pp.title( "%0.2f"%( err ))
    theta = b.theta_proposal_rand(theta)
    pp.plot( b.observations/1000.0 )
    pp.plot(test_obs/1000.0)
    pp.axis("off")
  pp.show()
  