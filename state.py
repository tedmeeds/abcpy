
# =========================================================================== #
#
#  BaseState: object that keeps a "state" of an MCMC chain or rejection 
#             sampling result.  Contains necessary functions: loglikelihood(), 
#             logprior(), rand(), etc.  Depends on the problem.
#
# =========================================================================== #
class BaseState( object ):
  def __init__( self, x, params ):
    self.nbr_sim_calls = 0
    self.x             = x
    self.params        = params
    
    self.load_params( self.params )
    
  def load_params( self, params ):
    raise NotImplementedError
    
  def loglikelihood( self, x = None ):
    raise NotImplementedError
    
  def logprior( self, x = None ):
    raise NotImplementedError
    
  def logproposal( self, q, x ):
    # from x to q logproposal
    raise NotImplementedError
  
  def discrepancy( self, x = None, obs = None ):
    # from x to q logproposal
    raise NotImplementedError
      
  def copy( self, other ):
    raise NotImplementedError
    
    