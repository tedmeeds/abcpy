from abcpy.abc_state import ABC_State

class ResponseModelState(ABC_State):
  def __init__( self, theta, response_model = None ):
    super(ResponseModelState, self).__init__(params)
    self.response_model = response_model
    
  def new( self ):
    return ResponseModelState( self.theta.copy(), self.response_model.copy() )
    
  def loglikelihood( self ):
    return self.response_model.loglikelihood()
    
  def loglikelihood_rand( self, M=1 ):
    return self.response_model.loglikelihood_rand(M)