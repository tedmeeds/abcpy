from abcpy.acquisition_model import AcquisitionModel

import numpy as np
import pylab as pp

class MetropolisHastingsErrorAcquisitionModel( AcquisitionModel ):
  
  def load_params( self, params ):
    hasattr( self, "surrogate" )
    
    self.n_steps_in_walk = params["nbr_steps"]
    pass
    
  def acquire(self, from_state, to_state, nbr_to_acquire, mh_model ):
    M             = mh_model.M
    logprior      = mh_model.logprior
    logproposal   = mh_model.logproposal
    proposal_rand = mh_model.proposal_rand
    
    current_logliks  = from_state.loglikelihood_rand( M )
    proposed_logliks = to_state.loglikelihood_rand( M )
    
    loglik_differences = proposed_logliks - current_logliks
    log_accs = mh_model.compute_log_acceptance_offset( from_state, to_state ) + loglik_differences
    
    I = pp.find( log_accs > 0 )
    log_accs[I] = 0
    accs     = np.exp(log_accs)
    median = np.median( accs )
    error0 = mh_model.metropolis_hastings_error( accs, median )
    
    errors = []
    qs     = []
    D = len(to_state.theta)
    for n in range( max(nbr_to_acquire,self.n_steps_in_walk)):
      
      # random walk from current location
      q_theta = proposal_rand( from_state.theta )
    
      q_state = from_state.new( q_theta, from_state.params )
      
      q_logliks = q_state.loglikelihood_rand( M )
    
      loglik_differences = q_logliks - current_logliks
      log_accs = mh_model.compute_log_acceptance_offset( from_state, q_state ) + loglik_differences
      
      I = pp.find( log_accs > 0 )
      log_accs[I] = 0
      accs     = np.exp(log_accs)
      median = np.median( accs )
      error1 = mh_model.metropolis_hastings_error( accs, median )
      
      log_accs = -mh_model.compute_log_acceptance_offset( from_state, q_state ) - loglik_differences
      
      I = pp.find( log_accs > 0 )
      log_accs[I] = 0
      accs     = np.exp(log_accs)
      median = np.median( accs )
      error2 = mh_model.metropolis_hastings_error( accs, median )
      
      #errors.append( max(error1,error2))
      errors.append( error1)
      qs.append(q_theta)
      
    smallest_to_biggest_errors = np.argsort( errors )
    
    
    # return thetas with highest errors  
    #print "MHE errors", errors
    thetas = []
    for n in range(nbr_to_acquire):
      thetas.append( qs[smallest_to_biggest_errors[-(n+1)]] )
      print "Acquire at ",   qs[smallest_to_biggest_errors[-(n+1)]], errors[smallest_to_biggest_errors[-(n+1)]], error0
    thetas = np.array( thetas )
    from_state.add( thetas.reshape( (nbr_to_acquire,D) ) )
    return thetas
    
    
    