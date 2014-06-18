from abcpy.response_models.gaussian_response_model import GaussianResponseModel # import SimulationResponseModel
from abcpy.helpers import mvn_logpdf, mvn_diagonal_logpdf, mvn_diagonal_logcdf
from abcpy.helpers import gaussian_logpdf, heavyside
import numpy as np
import pdb
  
class EpsilonHeavysideGaussianResponseModel( GaussianResponseModel ):
  
  def load_params( self, params ):
    self.likelihood_type = "logpdf"  # by default the likelihood is the density of observations under gaussian density
    self.diagonalize     = True     # by default, use full covariance
    self.epsilon         = 0.0
    
    self.down = 1
    self.up   = 0
    
    if params.has_key("epsilon"):
      self.epsilon = params["epsilon"]
      
    if params.has_key("min_epsilon"):
      self.min_epsilon = params["min_epsilon"] 
    else:
      self.min_epsilon = self.epsilon.copy()
      
    if params.has_key("cliff_at"):
      self.cliff_at = params["cliff_at"] 
    else:
      self.cliff_at = None
      
    # force user to decide if down-side epsilon or up-side epsilon 
    if params["direction"] == "down":
      self.direction = self.down
    elif params["direction"] == "up":
      self.direction = self.up
    else:
      self.direction = self.down
      
    self.epsilon_update_rule = None
    if params.has_key("epsilon_update_rule"):
      self.epsilon_update_rule = params["epsilon_update_rule"]
      self.epsilon_update_params = params["epsilon_update_params"]
      self.ema_decay = self.epsilon_update_params["decay"]
      
      if self.epsilon_update_params.has_key("ema"):
        self.ema     = self.epsilon_update_params["ema"]
        self.ema_var = self.epsilon_update_params["ema_var"]
        pdb.set_trace()
      else:
        self.ema     = self.epsilon
        self.ema_var = self.epsilon
      self.quantile = self.epsilon_update_params["quantile"]
   
  def new( self, params ):   
    m = EpsilonHeavysideGaussianResponseModel( params )
    return m
    
  def is_empty( self ):
    if len(self.pseudo_statistics) == 0:
      return True
    else:
      return False
        
  def add( self, thetas, pseudo_statistics, observation_statistics ):
    if len( self.pseudo_statistics ) == 0:
      self.pseudo_statistics = pseudo_statistics.copy()
      self.thetas = thetas.copy()
    else:
      self.pseudo_statistics = np.vstack( (self.pseudo_statistics, pseudo_statistics) )
      self.thetas = np.vstack( (self.thetas, thetas) )
      
    self.update()
       
  def update( self ):
    self.make_estimators( )
    
  def make_estimators( self, theta = None ):
    pseudo_statistics = self.pseudo_statistics
    
    # ignore thetas, observation_statistics
    
    # compute mean, stats_cov, mean_cov
    S,J = pseudo_statistics.shape
    assert S > 1, "must have at least 2 simulations"
    
    self.pstats_mean     = pseudo_statistics.mean(0)
    # d = pseudo_statistics - self.pstats_mean 
#     self.pstats_sumsq    = np.dot( d.T, d )
#     self.pstats_cov      = self.pstats_sumsq / (S-1) + self.epsilon*np.eye(J)
#     self.pstats_mean_cov = self.pstats_cov / S
#     self.pstats_icov     = np.linalg.inv( self.pstats_cov )
#     self.pstats_logdet   = -np.log( np.linalg.det( self.pstats_cov ) )  # neg cause of mvn parameterization
#     
#     # TODO: other options are to put priors
#     
#     # zero-out the off-diagonals;  make the statistics independent.
#     if self.diagonalize:
#       self.pstats_stddevs      = np.sqrt( np.diag( self.pstats_cov) )
#       self.pstats_mean_stddevs = np.sqrt( np.diag( self.pstats_mean_cov) )

  def update_post_mh( self, observation_group, simulation_statistics, params ):
    #print "before", self.ema, self.epsilon
    if self.epsilon_update_rule is not None:
      for ss in simulation_statistics:
        diff = self.discrepancy( observation_group.ystar, ss ) - self.ema
        incr = self.ema_decay*diff
        self.ema += self.ema_decay*diff
        self.ema_var = (1.0 - self.ema_decay) * (self.ema_var + diff * incr)
      
      #params["epsilon_update_params"]["ema_var"] = self.ema_var
      #params["epsilon_update_params"]["ema"] = self.ema
      epsilon = self.quantile*self.ema #np.sqrt(self.ema_var) #- np.sqrt(self.ema_var)
      #epsilon = self.ema - self.quantile * np.sqrt(self.ema_var) #- np.sqrt(self.ema_var)
      #print observation_group.ystar, ss, self.epsilon, self.discrepancy( observation_group.ystar, ss ), self.ema, self.ema_var
      for i in range(len(self.epsilon)):
        oo=epsilon.copy()
        pop=self.epsilon.copy()
        #print "epsilon[i], self.min_epsilon[i]:  ", epsilon[i], self.min_epsilon[i]
        self.epsilon[i] = max( epsilon[i], self.min_epsilon[i])
        #print "epsilon[i]  ",  epsilon[i]
        #self.epsilon[i] = min( self.epsilon[i], epsilon[i] )
        
        if self.epsilon[i] < self.min_epsilon[i]:
          pdb.set_trace()
      print "self.epsilon  ", self.epsilon, self.ema
       
  def loglikelihood( self, theta, observations ):
    self.make_estimators( theta )
    
    return self.loglikelihood_kernel( observations, self.pstats_mean)
    
  def loglikelihood_rand( self, theta, observations, N = 1 ):
    self.make_estimators( theta )
    
    random_logliks = self.loglikelihood(theta, observations)*np.ones( N )
      
    return random_logliks
    
  def discrepancy( self, ystar, y ):  
    J = len(y)
    d = y - ystar
    
    if self.direction == self.up:
      d *= -1
      
    N = 1
    # assume every observation is outside of tube
    disc = np.zeros( J )
  
    for j in range(J):
      h = heavyside( d[j] )
      if h > 0.5:
        disc[j] = d[j]
    return disc    
    
  def loglikelihood_kernel( self, observation_statistics, pseudo_statistics ):
    # start with the difference d =  y - y_star
    J = len(pseudo_statistics)
    d = pseudo_statistics - observation_statistics
    
    if self.direction == self.up:
      d *= -1
    
    N = 1
    # assume every observation is outside of tube
    loglikelihood = np.zeros( J )
  
    for j in range(J):
      h = heavyside( d[j] )
      if h > 0.5:
        loglikelihood[j] = -0.5*pow( d[j]/self.epsilon[j], 2 )
        
      if self.cliff_at is not None:
        if pseudo_statistics[j] == self.cliff_at[j]:
          loglikelihood[j] = -np.inf
    #pdb.set_trace()
    return np.sum(loglikelihood)