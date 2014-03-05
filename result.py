
def SimulationResult( object ):
  def __init__( self, theta, sim_output, stats ):
    self.theta      = theta
    self.sim_output = sim_output
    self.stats      = stats
    
# --------------------------------------------------------------------------- #
# Result: an object for containing results from an ABC algorithm, notably:
#
# -- simulation outputs
# -- 
# --------------------------------------------------------------------------- #
class SamplingResult( object ):
  def __init__(self, params):
    self.nbr_accepts     = 0
    self.nbr_sim_calls   = 0
    self.acceptance_rate = 0.0
    self.thetas      = []
    self.sim_outputs = []
    self.stats       = []
    self.discrepancy = []
    self.likelihood  = []
    self.acceptances = []
    
    self.rejected_thetas      = []
    self.rejected_sim_outputs = []
    self.rejected_stats       = []
    self.rejected_discrepancy = []
    self.rejected_likelihood  = []
    
    if params.has_key( "keep_discrepancy"):
      self.keep_discrepancy = params["keep_discrepancy"]
      
    if params.has_key( "keep_likelihood"):
      self.keep_likelihood = params["keep_likelihood"]
      
    if params.has_key( "keep_stats"):
      self.keep_stats = params["keep_stats"]
      
    if params.has_key( "keep_outputs"):
      self.keep_outputs = params["keep_outputs"]
      
    if params.has_key( "keep_rejections"):
      self.keep_rejections = params["keep_rejections"]
      
  def add_sim_result( self, accepted, theta, sim_outputs, stats, discrepancies = None,  ):
    pass
    