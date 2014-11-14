from abcpy.response_kernels.epsilon_tube                  import EpsilonTubeResponseKernel
from abcpy.response_kernels.epsilon_gaussian              import EpsilonGaussianResponseKernel
from abcpy.response_kernels.epsilon_heavyside_gaussian    import EpsilonHeavysideGaussianResponseKernel
from abcpy.response_kernels.epsilon_heavyside_exponential import EpsilonHeavysideExponentialResponseKernel
from abcpy.response_kernels.epsilon_quadratic_interval    import EpsilonQuadraticIntervalResponseKernel

from abcpy.response_models.gaussian_response_model       import GaussianResponseModel
from abcpy.response_models.epsilon_heavyside_gaussian    import EpsilonHeavysideGaussianResponseModel
from abcpy.response_models.epsilon_heavyside_exponential import EpsilonHeavysideExponentialResponseModel
from abcpy.response_models.surrogate_response_model      import SurrogateResponseModel
#from abcpy.response_models.kernel_density_response_model import KernelDensityResponseModel


def create_response( modeling_approach, response_type, response_params ):
  if modeling_approach == "kernel":
    if response_type == "gaussian":
      response = EpsilonGaussianResponseKernel( response_params )
    elif response_type == "tube":
      response = EpsilonTubeResponseKernel( response_params )
    elif response_type == "heavyside_gaussian":
      response = EpsilonHeavysideGaussianResponseKernel( response_params )
    elif response_type == "heavyside_exponential":
      response = EpsilonHeavysideExponentialResponseKernel( response_params )
    elif response_type == "quadratic_interval":
      response = EpsilonQuadraticIntervalResponseKernel( response_params )
    else:
      raise NotImplementedError
  elif modeling_approach == "local_model":
    if response_type == "gaussian":
      response = GaussianResponseModel( response_params )
    elif response_type == "heavyside_gaussian":
      response = EpsilonHeavysideGaussianResponseModel( response_params )
    elif response_type == "heavyside_exponential":
      response = EpsilonHeavysideExponentialResponseModel( response_params )
    else:
      raise NotImplementedError
  elif modeling_approach == "global_model":
    if response_type == "surrogate": 
        response = SurrogateResponseModel( response_params )
    else:
      raise NotImplementedError
  return response