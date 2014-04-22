class BaseSurrogate( object ):
  def __init__( self, params ):
    self.params = params
    self.load_params( params )
    
  def load_params( self, params ):
    raise NotImplementedError
  
  def make_estimators( self, theta = None ):
    # in future, may want to memoize conditional distributions
    pass
      
  def logpdf( self, theta, observations ):
    raise NotImplementedError
      
  def logcdf( self, theta, observations ):
    raise NotImplementedError
    
  def logcdfcomplement( self, theta, observations ):
    raise NotImplementedError
      
  def logpdf_rand( self, theta, observations, N = 1 ):
    raise NotImplementedError
    
  def logcdf_rand( self, theta, observations, N = 1 ):
    raise NotImplementedError
    
  def logcdfcomplement_rand( self, theta, observations, N = 1 ):
    raise NotImplementedError
