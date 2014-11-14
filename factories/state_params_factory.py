def scrape_params_from_problem( problem, S = 1 ):
  state_params = {}
  state_params["S"]                      = S
  state_params["observation_statistics"] = problem.get_obs_statistics()
  state_params["observation_groups"]     = problem.get_obs_groups()
  state_params["theta_prior_rand_func"]  = problem.theta_prior_rand
  state_params["simulation_function"]    = problem.simulation_function
  state_params["statistics_function"]    = problem.statistics_function
  
  return state_params