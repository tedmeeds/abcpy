
class AcquisitionModel( object ):
  def __init__( self, params ):
    self.params = params
    self.load_params( params )
    
  def load_params( self, params ):
    raise NotImplementedError
    
  def acquire( self, theta_from, theta_to, nbr_to_acquire ):
    raise NotImplementedError