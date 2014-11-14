def scrape_params_from_problem( problem, type="mh",is_marginal = True, nbr_samples = 1 ):
  mcmc_params = {}
  mcmc_params["type"]              = "mh"
  mcmc_params["priorrand"]         = problem.theta_prior_rand
  mcmc_params["logprior"]          = problem.theta_prior_logpdf
  mcmc_params["proposal_rand"]     = problem.theta_proposal_rand
  mcmc_params["logproposal"]       = problem.theta_proposal_logpdf
  mcmc_params["is_marginal"]       = is_marginal
  mcmc_params["nbr_samples"]       = nbr_samples
  
  return mcmc_params
