import pdb
from metropolis_hastings_model import BaseMetropolisHastingsModel

class SGLD_MetropolisHastingsModel( BaseMetropolisHastingsModel ):
  def __init__( self, params ):
    self.params = params
    self.load_params( params )
    self.current  = None
    self.proposed = None
    self.recorder = None
    self.nbr_sim_calls           = 0
    self.nbr_sim_calls_this_iter = 0
  
  def reset_nbr_sim_calls_this_iter(self):
    self.nbr_sim_calls_this_iter = 0
    if self.current is not None:
      self.current.reset_nbr_sim_calls_this_iter()
    if self.proposed is not None:
      self.proposed.reset_nbr_sim_calls_this_iter()
  
  def get_nbr_sim_calls_this_iter(self):
    self.nbr_sim_calls_this_iter = 0
    if self.current is not None:
      self.nbr_sim_calls_this_iter += self.current.nbr_sim_calls_this_iter
    if self.proposed is not None:
      self.nbr_sim_calls_this_iter += self.proposed.nbr_sim_calls_this_iter
    return self.nbr_sim_calls_this_iter 
  
  def describe_states(self):
    return ""
      
  def load_params( self, params ):
    self.logprior      = params["logprior"]
    self.logproposal   = params["logproposal"]
    self.proposal_rand = params["proposal_rand"]
    self.is_marginal   = params["is_marginal"]
  
  def set_recorder( self, recorder ):
    self.recorder = recorder 
   
  def propose_state( self ):
    q_state = self.current.new( self.proposal_rand( self.current.theta ), self.current.params ) 
    self.set_proposed_state( q_state )
    #pdb.set_trace()
       
  def set_proposed_state( self, proposed_state ):
    self.proposed = proposed_state
    
  def set_current_state( self, current_state ):
    #pdb.set_trace()
    self.current = current_state
  
  def update_current( self ):
    # for marginal sampler, we need to re-run the simulation at the current location
    if self.is_marginal:
      state = self.current.new( self.current.theta, self.current.params )  
      self.set_current_state( state )
    else:
      self.current.reset_nbr_sim_calls_this_iter()
    #pdb.set_trace()
   
  def stay_in_current_state( self ):
    #print "staying in current", self.get_nbr_sim_calls_this_iter()
    if self.recorder is not None:
      self.recorder.record_state( self.current, self.get_nbr_sim_calls_this_iter(), accepted = False )
      
    self.current.update_post_mh()   
      
  def move_to_proposed_state( self ):
    #pdb.set_trace()
    n = self.get_nbr_sim_calls_this_iter()
    self.proposed.update_post_mh()  
    self.set_current_state( self.proposed )
     
    #pdb.set_trace()
    if self.recorder is not None:
      self.recorder.record_state( self.current, n, accepted = True ) 
  
  def log_posterior(self):
    return self.current.loglikelihood() + self.logprior(self.current.theta)
      
  def log_acceptance( self ):
    q_state = self.proposed
    state   = self.current
    
    theta   = state.theta
    q_theta = q_state.theta
    
    # simulation -> outputs -> statistics -> loglikelihood 
    q_loglik   = q_state.loglikelihood()
    
    # prior log-density
    q_logprior = self.logprior( q_theta )
    
    # likelihood only computed once (state knows has already been computed)
    theta_loglik   = state.loglikelihood()
    theta_logprior = self.logprior( theta )
    
    # log-density of proposals
    q_to_theta_logproposal = self.logproposal( theta, q_theta )  
    theta_to_q_logproposal = self.logproposal( q_theta, theta )
    
    # Metropolis-Hastings acceptance log-probability and probability
    log_acc = min(0.0, q_loglik - theta_loglik + \
                       q_logprior - theta_logprior + \
                       q_to_theta_logproposal - theta_to_q_logproposal \
                 )
    #print q_loglik,theta_loglik
                 
    return log_acc