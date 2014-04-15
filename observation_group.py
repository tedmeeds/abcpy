
# -- subset of statistics canbe treated and modeled differently.  
# -- some observations might be iids vectors
# -- some observations may be sets of constraints

class ObservationGroup( object ):
  def __init__( self, stat_ids, observation_statistics, params ):
    self.ids = stat_ids # which statistics are used for this group
    self.observation_statistics = observation_statistics # subset of all observations
    self.ystar = observation_statistics
    self.params = params