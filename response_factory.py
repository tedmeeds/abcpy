from abcpy.response_kernels.epsilon_tube     import EpsilonTubeResponseKernel
from abcpy.response_kernels.epsilon_gaussian import EpsilonGaussianResponseKernel
from abcpy.response_kernels.epsilon_heavyside_gaussian import EpsilonHeavysideGaussianResponseKernel
from abcpy.response_kernels.epsilon_heavyside_exponential import EpsilonHeavysideExponentialResponseKernel
from abcpy.response_kernels.epsilon_quadratic_interval import EpsilonQuadraticIntervalResponseKernel

from abcpy.response_models.gaussian_response_model import GaussianResponseModel
from abcpy.response_models.epsilon_heavyside_gaussian import EpsilonHeavysideGaussianResponseModel
from abcpy.response_models.epsilon_heavyside_exponential import EpsilonHeavysideExponentialResponseModel
from abcpy.response_models.surrogate_response_model import SurrogateResponseModel
#from abcpy.response_models.kernel_density_response_model import KernelDensityResponseModel


def response_factory( response_type, response_params ):
  if response_type == "gaussian_kernel":
    response = EpsilonGaussianResponseKernel( response_params )
  elif response_type == "gaussian_model":
    response = GaussianResponseModel( response_params )
  elif response_type == "heavyside_gaussian_kernel":
    response = EpsilonHeavysideGaussianResponseKernel( response_params )
  elif response_type == "heavyside_gaussian_model":
    response = EpsilonHeavysideGaussianResponseModel( response_params )
  elif response_type == "heavyside_exponential_kernel":
    response = EpsilonHeavysideExponentialResponseKernel( response_params )
  elif response_type == "heavyside_exponential_model":
    response = EpsilonHeavysideExponentialResponseModel( response_params )
  elif response_type == "surrogate_response_model":
    response = SurrogateResponseModel( response_params )
  elif response_type == "quadratic_interval_kernel":
    response = EpsilonQuadraticIntervalResponseKernel( response_params )
  elif response_type == "tube":
    response = EpsilonTubeResponseKernel( response_params )
  else:
    print response_type, response_params
    raise NotImplementedError
  
  return response