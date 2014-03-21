from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel
import numpy as np
import pylab as pp
import pdb

def conditional_metropolis_hastings_error( alphas, tau, u ):
  # alphas:  random samples of acceptance probabiltiies
  # tau:     threshold, typically the empirical median of alphas
  
  # can only have errors on one "side" of tau
  if u < tau:
    # an accept
    # error is number of alphas less than u
    mistakes = len( pp.find( alphas < u ) )
  else:
    # a reject
    # error is number of alphas greater than u
    mistakes = len( pp.find( alphas >= u ) )
    
  return float(mistakes)/float(len(alphas))
  
def unconditional_metropolis_hastings_error( alphas, tau ):
  # alphas:  random samples of acceptance probabiltiies
  # tau:     threshold, typically the empirical median of alphas
  err = 0.0
  u_stream = np.linspace( 0, 1, 100 )
  #u_stream = np.random.rand( 100 )
  
  # "integrate" over u draws -- NB this is pretty inefficient way of doing it
  for u in u_stream:
    err += conditional_metropolis_hastings_error( alphas, tau, u )
  err /= len( u_stream )
  
  return err
  
class AdaptiveMetropolisHastingsModel( BaseMetropolisHastingsModel ):
    
  def load_params( self, params ):
    self.xi            = params["xi"]
    self.M             = params["M"]
    self.deltaS        = params["deltaS"]
    self.max_nbr_tries = params["max_nbr_tries"] 
    self.errors        = []
    
  def metropolis_hastings_error( self, acceptance_values,  u = None ):
    self.median = np.median( acceptance_values )
    if u is None:
      return unconditional_metropolis_hastings_error( acceptance_values, self.median )
    else:
      return conditional_metropolis_hastings_error( acceptance_values, self.median, u )
  
  def compute_log_acceptance_offset( self ):
    # prior log-density
    q_logprior     = self.proposed.logprior( self.proposed.theta )
    theta_logprior = self.current.logprior( self.current.theta )
  
    # log-density of proposals
    q_to_theta_logproposal = self.proposed.logproposal( self.current.theta, self.proposed.theta )  
    theta_to_q_logproposal = self.current.logproposal( self.proposed.theta, self.current.theta )
      
    # this quantity is constant, the log-likelihood varies
    return q_logprior - theta_logprior + q_to_theta_logproposal - theta_to_q_logproposal
 
  def loglik_differences_rand( self, M ):   
      proposed_logliks = self.proposed.loglikelihood_rand( M )
      current_logliks  = self.current.loglikelihood_rand( M )
      return proposed_logliks-current_logliks
      
  def log_acceptance( self, u = None ):
    # this quantity is constant, the log-likelihood varies
    self.log_acceptance_offset = self.compute_log_acceptance_offset()
    
    self.error = np.inf
    nbr_tries = 0
    while (self.error > self.xi) and (nbr_tries < self.max_nbr_tries):
      loglik_differences = self.loglik_differences_rand( self.M )
    
      self.log_accs = self.log_acceptance_offset + loglik_differences
      
      
      I = pp.find( self.log_accs > 0 )
      self.log_accs[I] = 0
      self.accs     = np.exp(self.log_accs)
      self.error = self.metropolis_hastings_error( self.accs, u )
      
      #pdb.set_trace()
      if nbr_tries == 0:
        was_error = self.error
      # print "  from ", self.proposed.theta
      # print "  to ", self.current.theta
      # print "  has error = ",self.error
      
      
      if nbr_tries < self.max_nbr_tries:
        if self.error > self.xi:
          
          print "\t",nbr_tries, self.median, "  ","error > xi: ",self.error, self.describe_states()
          self.acquire_points()
          nbr_tries += 1
    
    self.errors.append( self.error )
    if nbr_tries > 0:    
      print "\t",nbr_tries, "median = ", self.median, "  ","error from: ",was_error, " to ", self.error, self.describe_states()
    # Metropolis-Hastings acceptance log-probability and probability
    if self.median > 0:
      return np.log( self.median )
    else:
      return -np.inf
    
  def acquire_points( self ):
    if np.random.rand() < 0.5:
      self.proposed.acquire( self.deltaS )
    else:
      self.current.acquire( self.deltaS )