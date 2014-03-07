

class BaseMetropolisHastingsModel( object ):
  def __init__( self, params ):
    self.params = params
    
  def set_proposed_state( self, proposed_state ):
    self.proposed_state = proposed_state
    
  def set_current_state( self, current_state ):
    self.current_state = current_state
    
  def log_acceptance( self ):
    q_state = self.proposed_state
    state   = self.current_state
    
    theta   = state.theta
    q_theta = q_state.theta
    
    # simulation -> outputs -> statistics -> loglikelihood 
    q_loglik   = q_state.loglikelihood()
    
    # prior log-density
    q_logprior = q_state.logprior( q_theta )
    
    # likelihood only computed once (state knows has already been computed)
    theta_loglik   = state.loglikelihood()
    theta_logprior = state.logprior( theta )
    
    # log-density of proposals
    q_to_theta_logproposal = q_state.logproposal( theta, q_theta )  
    theta_to_q_logproposal = state.logproposal( q_theta, theta )
    
    # Metropolis-Hastings acceptance log-probability and probability
    log_acc = min(0.0, q_loglik - theta_loglik + \
                       q_logprior - theta_logprior + \
                       q_to_theta_logproposal - theta_to_q_logproposal \
                 )
                 
    return log_acc