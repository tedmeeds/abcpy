
# =========================================================================== #
#
#  BaseState: object that keeps a "state" of an MCMC chain or rejection 
#             sampling result.  Contains necessary functions: loglikelihood(), 
#             logprior(), rand(), etc.  Depends on the problem.
#
# =========================================================================== #
class BaseState( object ):
  def __init__( self, theta, params ):
    self.nbr_sim_calls = 0
    self.theta         = theta
    self.params        = params
    
    self.load_params( self.params )
    
  def load_params( self, params ):
    raise NotImplementedError
    
  def loglikelihood( self, theta = None ):
    raise NotImplementedError
    
  def logprior( self, theta = None ):
    raise NotImplementedError
    
  def logproposal( self, q, theta ):
    # from x to q logproposal
    raise NotImplementedError
  
  def discrepancy( self, theta = None, obs = None ):
    # from x to q logproposal
    raise NotImplementedError
      
  def copy( self, other ):
    raise NotImplementedError
    
    