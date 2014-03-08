from abcpy.models.metropolis_hastings_model import BaseMetropolisHastingsModel

def unconditional_metropolis_hastings_error( alphas, tau, u ):
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
  u_stream = np.linspace( 0, 1, 0.001 )
  
  # "integrate" over u draws -- NB this is pretty inefficient way of doing it
  for u in u_stream:
    err += conditional_metropolis_hastings_error( alphas, tau, u )
  err /= len( u_stream )
  
  return err
  
class AdaptiveSyntheticLikelihoodModel( BaseMetropolisHastingsModel ):
    
  def load_params( self, params ):
    self.xi            = params["xi"]
    self.M             = params["M"]
    self.deltaS        = params["deltaS"]
    self.max_nbr_tries = params["max_nbr_tries"] 
    
  def metropolis_hastings_error( self, acceptance_values,  u = None ):
    self.median = np.median( acceptance_values )
    if u is None:
      return unconditional_metropolis_hastings_error( acceptance_values, self.median )
    else:
      return conditional_metropolis_hastings_error( acceptance_values, self.median, u )
  
  def self.compute_log_acceptance_offset( self ):
    # prior log-density
    q_logprior     = self.proposed_state.logprior( self.proposed_state.q_theta )
    theta_logprior = self.current_state.logprior( self.current_state.theta )
  
    # log-density of proposals
    q_to_theta_logproposal = self.proposed_state.logproposal( self.current_state.theta, self.proposed_state.theta )  
    theta_to_q_logproposal = self.current_state.logproposal( q_theta, self.current_state.theta )
      
    # this quantity is constant, the log-likelihood varies
    return q_logprior - theta_logprior + q_to_theta_logproposal - theta_to_q_logproposal
    
  def log_acceptance( self, u = None ):
    # this quantity is constant, the log-likelihood varies
    self.log_acceptance_offset = self.compute_log_acceptance_offset()
    
    error = np.inf
    nbr_tries = 0
    while (error > self.xi) and (nbr_tries < self.max_nbr_tries):
      proposed_logliks = self.proposed_state.loglikelihood_rand( self.M )
      current_logliks  = self.current_state.loglikelihood_rand( self.M )
    
      self.log_accs = self.log_acceptance_offset + proposed_logliks - current_logliks
      self.accs     = np.exp(self.log_accs)
      
      I = pp.find( log_accs > 0 )
      log_accs[I] = 0
      
      self.error = self.metropolis_hastings_error( self.accs, u )
      nbr_tries += 1
      
      if nbr_tries < self.max_nbr_tries:
        if self.error > self.xi:
          self.acquire_points()
        
    # Metropolis-Hastings acceptance log-probability and probability
    return np.log( self.median )
    
  def acquire_points( self ):
    assert False, "TODO"
    pass