from abcpy.models.metropolis_hastings_model import BaseMetropolisHastingsModel

class AdaptiveSyntheticLikelihoodModel( BaseMetropolisHastingsModel ):
    
  def load_params( self, params ):
    pass
    
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
    
    q_mu_stats     =  q_state.stats.mean(0)
    q_cov_stats    =  q_state.stats.cov()
    q_cov_mu_stats = q_cov_stats / q_state.S
    
    mu_stats     =  state.stats.mean(0)
    cov_stats    =  state.stats.cov()
    cov_mu_stats =  state.stats.cov() / state.S
    
    # Metropolis-Hastings acceptance log-probability and probability
    log_acc = min(0.0, q_loglik - theta_loglik + \
                       q_logprior - theta_logprior + \
                       q_to_theta_logproposal - theta_to_q_logproposal \
                 )
                 
    return log_acc