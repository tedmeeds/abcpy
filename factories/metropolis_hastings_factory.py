from abcpy.metropolis_hastings_models.metropolis_hastings_model import BaseMetropolisHastingsModel
from abcpy.metropolis_hastings_models.adaptive_metropolis_hastings_model import AdaptiveMetropolisHastingsModel
from abcpy.metropolis_hastings_models.sgld_model import SGLD_MetropolisHastingsModel

def create_mh_model( mcmc_params ):
  if mcmc_params["type"] == "mh":
    mh_model = BaseMetropolisHastingsModel( mcmc_params )
  elif mcmc_params["type"] == "amh":
    mh_model = AdaptiveMetropolisHastingsModel( mcmc_params )
  elif mcmc_params["type"] == "sgldmh":
    mh_model = SGLD_MetropolisHastingsModel( mcmc_params )
  else:
    raise NotImplementedError
  return mh_model