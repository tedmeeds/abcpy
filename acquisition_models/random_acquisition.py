from abcpy.acquisition_model import AcquisitionModel

import numpy as np

class RandomAcquisitionModel( AcquisitionModel ):
  
  def load_params( self, params ):
    pass
    
  def acquire(self, from_state, to_state, nbr_to_acquire ):
    thetas = []
    for n in range(nbr_to_acquire):
      if np.random.randn() < 0:
        thetas.append( from_state.theta )
        from_state.acquire()
      else:
        thetas.append( to_state.theta )
        to_state.acquire()
        
    thetas = np.array( thetas )
    return thetas