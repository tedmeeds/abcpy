import os
from abcpy.recorders.new_recorder import Recorder

def if_file_init_it( filename ):
  if filename is not None:
    d = os.path.dirname(filename)
    if os.path.exists( d ) is False: 
      os.path.os.mkdir( d)
    fptr = open(filename,'w+');  fptr.close();
    
def create_recorder( params ):
  write_each           = False
  force_directory_init = True
  state_theta_log_file = None
  state_stats_log_file = None
  state_logpost_log_file = None
  sim_theta_log_file   = None
  sim_stats_log_file   = None
  
  #mcmc_dir = params["mcmc_dir"]  # directory where the states of the mc are recorded
  #sim_dir  = params["sim_dir"]   # directory where the log of all simulations are recorded
  
  # where to record the thetas (input parameters) -- state of mcmc / rejection
  if params.has_key("state_theta_log_filename"):
    state_theta_log_file = params["state_theta_log_filename"]
  # where to record the stats (per theta )  -- state of mcmcm / rejection
  if params.has_key("state_stats_log_filename"):
    state_stats_log_file = params["state_stats_log_filename"]
  if params.has_key("state_logpost_log_file"):
    state_logpost_log_file = params["state_logpost_log_file"]
  # where to record the simulation results (ie all simulations)
  if params.has_key("sim_theta_log_filename"):
    sim_theta_log_file   = params["sim_theta_log_filename"]
  if params.has_key("sim_stats_log_filename"):
    sim_stats_log_file   = params["sim_stats_log_filename"]
  
  if params.has_key("force_directory_init"):
    force_directory_init = params["force_directory_init"]
  
  if force_directory_init:
    if_file_init_it( state_theta_log_file )
    if_file_init_it( state_stats_log_file )
    if_file_init_it( state_logpost_log_file )
    if_file_init_it( sim_theta_log_file )
    if_file_init_it( sim_stats_log_file )

  if params.has_key("write_each"):
    write_each = params["write_each"]
    
  recorder = Recorder( state_theta_log_file, state_stats_log_file, state_logpost_log_file, \
                       sim_theta_log_file, sim_stats_log_file, \
                       write_each )
  
  return recorder
  