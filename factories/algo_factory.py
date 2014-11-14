from abcpy.factories.response_factory import create_response
from abcpy.factories.metropolis_hastings_factory import create_mh_model

from abcpy.states.discrepancy_state                             import DiscrepancyState 
from abcpy.states.kernel_based_state                            import KernelState 
from abcpy.states.response_model_state                          import ResponseModelState

from abcpy.algos.rejection     import abc_rejection    
from abcpy.algos.mcmc          import abc_mcmc  as simple_abc_mcmc
from abcpy.algos.model_mcmc    import abc_mcmc  as model_abc_mcmc
    
    
def create_algo_and_state( params ):
  modeling_approach = params["modeling_approach"]
  ogs               = params["observation_groups"]
  state_params      = params["state_params"]
  algorithm         = params["algorithm"]

  # ------------------------------------------------------------------
  # 1) for all observation groups, construct correct kernel/model/etc
  # ------------------------------------------------------------------
  response_groups = []
  for og in params["observation_groups"]:
    response = create_response( modeling_approach, og.params["response_type"], og.params["response_params"])
    response_groups.append(response)
    
  state_params["response_groups"] = response_groups
  
  # ---------------------------------------------------------------
  # 2) determine state type used in inference
  # ---------------------------------------------------------------
  modeling_approach = params["modeling_approach"]
  is_kernel    = True
  is_surrogate = False
  if modeling_approach == "kernel":
    # eg epsilon_tube, kernel_epsilon, etc
    is_kernel    = True
    is_surrogate = False
    
    if algorithm == "rejection":
      state = DiscrepancyState( state_params )
    else:
      state = KernelState( state_params )
      
  elif modeling_approach == "local_model":
    # eg synthetic likelihood, kde, etc
    assert algorithm != "rejection", "local models do not work with rejection sampler"
    is_kernel    = False
    is_surrogate = False
    state = ResponseModelState( state_params )
    
  elif modeling_approach == "global_model":
    # eg surrogates: global KDE, GP surrogates
    assert algorithm != "rejection", "global models do not work with rejection sampler"
    is_kernel    = False
    is_surrogate = True
    state = ResponseModelState( state_params )
    
  else:
    raise NotImplementedError
    
  # ---------------------------------------------------------------
  # 3) determine algorithm, MH, MH with model, rejection, etc
  # ---------------------------------------------------------------
  if algorithm == "rejection":
    algo = abc_rejection
    return algo, state
  elif algorithm == "simple_mcmc":
    algo = simple_abc_mcmc
    return algo, state
  elif algorithm == "model_mcmc":
    mh_model = create_mh_model( params["mcmc_params"] )
    #mh_model.set_current_state( state )
    #mh_model.set_recorder( recorder )
    algo = model_abc_mcmc
    return algo, mh_model, state
  else:
    raise NotImplementedError
    
  