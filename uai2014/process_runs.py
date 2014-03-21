import numpy as np
import scipy as sp
import pylab as pp
import cPickle

class UaiSummary(object):
  def __init__( self, problem_name, experiment_name ):
    self.problem_name = problem_name
    self.experiment_name = experiment_name
    
  def add_thetas( self, the_list ):
    self.thetas = np.array( the_list ).T
    
  def add_stats( self, the_list ):
    self.stats = np.array( the_list ).T
    
  def add_sims( self, the_list ):
    self.sims = np.array( the_list ).T
    
    # for sims want the cumsum
    self.sims_cumsum      = np.cumsum( self.sims, 0 )
    self.sims_cumsum_mean = self.sims_cumsum.mean(1)
    self.sims_cumsum_std  = self.sims_cumsum.std(1)

    self.total_sims      = self.sims_cumsum[-1]
    self.mean_total_sims = self.sims_cumsum_mean[-1]
    
    
  def add_accs( self, the_list ):
    self.accs = np.array( the_list ).T
  
    # for sims want the cumsum
    self.accs_cumsum      = np.cumsum( self.accs, 0 )
    self.accs_cumsum_mean = self.accs_cumsum.mean(1)
    self.accs_cumsum_std  = self.accs_cumsum.std(1)

    self.total_accs      = self.accs_cumsum[-1]
    self.mean_total_accs = self.accs_cumsum_mean[-1]

def collect_sl_based( kinds, styles, Ss, problem_name, ds = None ):
    
  runs_dir      = "./uai2014/runs"
  summaries_dir = "./uai2014/summaries"
  #kinds        = ["abc_mcmc_marginal","abc_mcmc_pseudo"]
  #Ss           = [1,2,10]
  #epsilons     = [5.0,2.0,1.0,0.1,0.05,0.01]
  #repeats      = 10
  #problem_name = "exponential"
  for kind in kinds:
    for S in Ss:
      save_dir = "%s/%s/%s_s%d/"%(runs_dir,problem_name,kind,S)
      for style in styles:
        #epsilon_string = "eps" + str(epsilon).replace(".","p")
        
        if ds is None:
          experiment_name = "%s_%s_s%d"%(kind,style,S) 
        else:
          experiment_name = "%s_%s_s%d_ds%s"%(kind,style,S,ds)
          
      
        print "WORKING EXPERIMENT: %s"%(experiment_name)
      
        THETAS = []
        SIMS   = []
        ACCS   = []
        for repeat in range(repeats):
          out_name = save_dir + style + "_" + "repeat%d"%(repeat+1)
      
          print "\t PROCESSING..." + out_name
          accs    = np.loadtxt( out_name + "_acceptances.txt" )
          sims    = np.loadtxt( out_name + "_sims.txt" )
          thetas  = np.loadtxt( out_name + "_thetas.txt" )
        
          THETAS.append(thetas)
          ACCS.append(accs)
          SIMS.append(sims)
      
        uai = UaiSummary( problem_name, experiment_name)
        uai.style = style
        uai.add_thetas(THETAS)
        uai.add_sims(SIMS)
        uai.add_accs(ACCS)
    
        savefile = "%s/%s/%s.pkl"%(summaries_dir,problem_name,experiment_name)
        print "\t SAVING PICKLE...%s"%(savefile)
        cPickle.dump( uai, open( savefile, "w+") )
        print "\t ...complete"
        
def collect_epsilon_based( kinds, Ss, epsilons, problem_name, ds = None ):
    
  runs_dir      = "./uai2014/runs"
  summaries_dir = "./uai2014/summaries"
  kinds        = ["abc_mcmc_marginal","abc_mcmc_pseudo"]
  Ss           = [1,2,10]
  epsilons     = [5.0,2.0,1.0,0.1,0.05,0.01]
  repeats      = 10
  problem_name = "exponential"
  for kind in kinds:
    for S in Ss:
      if ds is None:
        save_dir = "%s/%s/%s_s%d/"%(runs_dir,problem_name,kind,S)
      else:
        save_dir = "%s/%s/%s_s%d_ds%d/"%(runs_dir,problem_name,kind,S,ds)
      for epsilon in epsilons:
        epsilon_string = "eps" + str(epsilon).replace(".","p")
      
        experiment_name = "%s_s%d_%s"%(kind,S,epsilon_string) 
      
        print "WORKING EXPERIMENT: %s"%(experiment_name)
      
        THETAS = []
        SIMS   = []
        ACCS   = []
        for repeat in range(repeats):
          out_name = save_dir + epsilon_string + "_" + "repeat%d"%(repeat+1)
      
          print "\t PROCESSING..." + out_name
          accs    = np.loadtxt( out_name + "_acceptances.txt" )
          sims    = np.loadtxt( out_name + "_sims.txt" )
          thetas  = np.loadtxt( out_name + "_thetas.txt" )
        
          THETAS.append(thetas)
          ACCS.append(accs)
          SIMS.append(sims)
      
        uai = UaiSummary( problem_name, experiment_name)
        uai.epsilon = epsilon
        uai.add_thetas(THETAS)
        uai.add_sims(SIMS)
        uai.add_accs(ACCS)
    
        savefile = "%s/%s/%s.pkl"%(summaries_dir,problem_name,experiment_name)
        print "\t SAVING PICKLE...%s"%(savefile)
        cPickle.dump( uai, open( savefile, "w+") )
        print "\t ...complete"

def collect_asl_based( kinds, styles, knob_values, Ss, problem_name, repeats, ds = None ):
    
  runs_dir      = "./uai2014/runs"
  summaries_dir = "./uai2014/summaries"
  #kinds        = ["abc_mcmc_marginal","abc_mcmc_pseudo"]
  #Ss           = [1,2,10]
  #epsilons     = [5.0,2.0,1.0,0.1,0.05,0.01]
  #repeats      = 10
  #problem_name = "exponential"
  for kind in kinds:
    for S in Ss:
      if ds is None:
        save_dir = "%s/%s/%s_s%d/"%(runs_dir,problem_name,kind,S)
      else:
        save_dir = "%s/%s/%s_s%d_ds%d/"%(runs_dir,problem_name,kind,S,ds)
      for epsilon in knob_values:
        for style in styles:
          epsilon_string = "xi" + str(epsilon).replace(".","p")
      
          experiment_name = "%s_s%d_%s"%(kind, S, epsilon_string) 
      
          print "WORKING EXPERIMENT: %s"%(experiment_name)
      
          THETAS = []
          SIMS   = []
          ACCS   = []
          for repeat in range(repeats):
            out_name = save_dir + epsilon_string + "_" + style + "_"+"repeat%d"%(repeat+1)
      
            print "\t PROCESSING..." + out_name
            accs    = np.loadtxt( out_name + "_acceptances.txt" )
            sims    = np.loadtxt( out_name + "_sims.txt" )
            thetas  = np.loadtxt( out_name + "_thetas.txt" )
        
            THETAS.append(thetas)
            ACCS.append(accs)
            SIMS.append(sims)
      
          uai = UaiSummary( problem_name, experiment_name)
          uai.epsilon = epsilon
          uai.add_thetas(THETAS)
          uai.add_sims(SIMS)
          uai.add_accs(ACCS)
    
          savefile = "%s/%s/%s.pkl"%(summaries_dir,problem_name,experiment_name)
          print "\t SAVING PICKLE...%s"%(savefile)
          cPickle.dump( uai, open( savefile, "w+") )
          print "\t ...complete"

def collect_gps_based( kinds, styles, knob_values, problem_name, repeats, ds = None ):
    
  runs_dir      = "./uai2014/runs"
  summaries_dir = "./uai2014/summaries"
  #kinds        = ["abc_mcmc_marginal","abc_mcmc_pseudo"]
  #Ss           = [1,2,10]
  #epsilons     = [5.0,2.0,1.0,0.1,0.05,0.01]
  #repeats      = 10
  #problem_name = "exponential"
  for kind in kinds:
    save_dir = "%s/%s/%s/"%(runs_dir,problem_name,kind)
    for epsilon in knob_values:
      for style in styles:
        epsilon_string = "xi" + str(epsilon).replace(".","p")
    
        experiment_name = "%s_%s"%(kind, epsilon_string) 
    
        print "WORKING EXPERIMENT: %s"%(experiment_name)
    
        THETAS = []
        SIMS   = []
        ACCS   = []
        for repeat in range(repeats):
          out_name = save_dir + epsilon_string + "_" + style + "_"+"repeat%d"%(repeat+1)
    
          print "\t PROCESSING..." + out_name
          accs    = np.loadtxt( out_name + "_acceptances.txt" )
          sims    = np.loadtxt( out_name + "_sims.txt" )
          thetas  = np.loadtxt( out_name + "_thetas.txt" )
      
          THETAS.append(thetas)
          ACCS.append(accs)
          SIMS.append(sims)
    
        uai = UaiSummary( problem_name, experiment_name)
        uai.epsilon = epsilon
        uai.add_thetas(THETAS)
        uai.add_sims(SIMS)
        uai.add_accs(ACCS)
  
        savefile = "%s/%s/%s.pkl"%(summaries_dir,problem_name,experiment_name)
        print "\t SAVING PICKLE...%s"%(savefile)
        cPickle.dump( uai, open( savefile, "w+") )
        print "\t ...complete"

def collect_rejection_based( kinds, knob_values, problem_name, repeats, ds = None ):
    
  runs_dir      = "./uai2014/runs"
  summaries_dir = "./uai2014/summaries"
  #kinds        = ["abc_mcmc_marginal","abc_mcmc_pseudo"]
  #Ss           = [1,2,10]
  #epsilons     = [5.0,2.0,1.0,0.1,0.05,0.01]
  #repeats      = 10
  #problem_name = "exponential"
  for kind in kinds:
    save_dir = "%s/%s/%s/"%(runs_dir,problem_name,kind)
    for epsilon in knob_values:
      epsilon_string = "eps" + str(epsilon).replace(".","p")
  
      experiment_name = "%s_%s"%(kind, epsilon_string) 
  
      print "WORKING EXPERIMENT: %s"%(experiment_name)
  
      THETAS = []
      SIMS   = []
      ACCS   = []
      for repeat in range(repeats):
        out_name = save_dir + epsilon_string + "_"+"repeat%d"%(repeat+1)
  
        print "\t PROCESSING..." + out_name
        accs    = np.loadtxt( out_name + "_acceptances.txt" )
        sims    = np.loadtxt( out_name + "_sims.txt" )
        thetas  = np.loadtxt( out_name + "_thetas.txt" )
    
        THETAS.append(thetas)
        ACCS.append(accs)
        SIMS.append(sims)
  
      uai = UaiSummary( problem_name, experiment_name)
      uai.epsilon = epsilon
      uai.add_thetas(THETAS)
      uai.add_sims(SIMS)
      uai.add_accs(ACCS)

      savefile = "%s/%s/%s.pkl"%(summaries_dir,problem_name,experiment_name)
      print "\t SAVING PICKLE...%s"%(savefile)
      cPickle.dump( uai, open( savefile, "w+") )
      print "\t ...complete"
                                          
if __name__ == "__main__":
  # ------------- #
  # EXPERIMENT 1  #
  # ------------- #
  kinds        = ["abc_mcmc_marginal","abc_mcmc_pseudo"]
  Ss           = [1,2,10]
  epsilons     = [5.0,2.0,1.0,0.1,0.05,0.01]
  repeats      = 10
  problem_name = "exponential"
  # uncomment to process:
  #collect_epsilon_based( kinds, Ss, epsilons, problem_name )  
  
  # ------------- #
  # EXPERIMENT 2  #
  # ------------- #
  kinds        = ["sl_marginal","sl_pseudo"]
  Ss           = [2,5,10,50]
  styles       = ["just_gaussian"]
  #epsilons     = [5.0,2.0,1.0,0.1,0.05,0.01]
  repeats      = 10
  problem_name = "exponential"
  # uncomment to process:
  #collect_sl_based( kinds, styles, Ss, problem_name )
  
  # ------------- #
  # EXPERIMENT 3  #
  # ------------- #
  kinds        = ["asl_pseudo"]
  Ss           = [2,5,10]
  ds           = 5
  styles       = ["just_gaussian"]
  knob         = ["xi"]
  knob_values  = [0.4,0.3,0.2,0.1]
  repeats      = 10
  problem_name = "exponential"
  # uncomment to process:
  #collect_asl_based( kinds, styles, knob_values, Ss, problem_name, repeats,ds = ds )
  
  # ------------- #
  # EXPERIMENT 4  #
  # ------------- #
  kinds        = ["gps"]
  Ss           = [2,5,10]
  ds           = None
  styles       = ["just_gaussian"]
  knob         = ["xi"]
  knob_values  = [0.4,0.3,0.2]
  repeats      = 10
  problem_name = "exponential"
  # uncomment to process:
  #collect_gps_based( kinds, styles, knob_values, problem_name, repeats  )
  
  # ------------- #
  # EXPERIMENT 5  #
  # ------------- #
  kinds        = ["rejection"]
  epsilons     = [0.5,0.1,0.05]
  repeats      = 10
  problem_name = "exponential"
  # uncomment to process:
  #collect_rejection_based( kinds, epsilons, problem_name, repeats )