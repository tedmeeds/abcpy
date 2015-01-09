from abcpy.problem import BaseProblem
from abcpy.observation_group import ObservationGroup

from abcpy.plotting import *
from abcpy.helpers import *
import numpy as np
import scipy as sp
import pylab as pp

import pdb

#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
#
#
#This file contains a Python version of Carl Rasmussen's Matlab-function 
#minimize.m
#
#minimize.m is copyright (C) 1999 - 2006, Carl Edward Rasmussen.
#Python adaptation by Roland Memisevic 2008.
#
#
#The following is the original copyright notice that comes with the 
#function minimize.m
#(from http://www.kyb.tuebingen.mpg.de/bs/people/carl/code/minimize/Copyright):
#
#
#"(C) Copyright 1999 - 2006, Carl Edward Rasmussen
#
#Permission is granted for anyone to copy, use, or modify these
#programs and accompanying documents for purposes of research or
#education, provided this copyright notice is retained, and note is
#made of any changes that have been made.
#
#These programs and documents are distributed without any warranty,
#express or implied.  As the programs were written for research
#purposes only, they have not been tested to the degree that would be
#advisable in any important application.  All use of these programs is
#entirely at the user's own risk."


"""minimize.py 

This module contains a function 'minimize' that performs unconstrained
gradient based optimization using nonlinear conjugate gradients. 

The function is a straightforward Python-translation of Carl Rasmussen's
Matlab-function minimize.m

"""


from numpy import dot, isinf, isnan, any, sqrt, isreal, real, nan, inf

def minimize(X, f, grad, args, maxnumlinesearch=None, maxnumfuneval=None, red=1.0, verbose=False):
    INT = 0.1;# don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0;              # extrapolate maximum 3 times the current step-size
    MAX = 20;                     # max 20 function evaluations per line search
    RATIO = 10;                                   # maximum allowed slope ratio
    SIG = 0.1;RHO = SIG/2;# SIG and RHO are the constants controlling the Wolfe-
    #Powell conditions. SIG is the maximum allowed absolute ratio between
    #previous and new slopes (derivatives in the search direction), thus setting
    #SIG to low (positive) values forces higher precision in the line-searches.
    #RHO is the minimum allowed fraction of the expected (from the slope at the
    #initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    #Tuning of SIG (depending on the nature of the function to be optimized) may
    #speed up the minimization; it is probably not worth playing much with RHO.

    SMALL = 10.**-16                    #minimize.m uses matlab's realmin 
    
    if maxnumlinesearch == None:
        if maxnumfuneval == None:
            raise "Specify maxnumlinesearch or maxnumfuneval"
        else:
            S = 'Function evaluation'
            length = maxnumfuneval
    else:
        if maxnumfuneval != None:
            raise "Specify either maxnumlinesearch or maxnumfuneval (not both)"
        else: 
            S = 'Linesearch'
            length = maxnumlinesearch

    i = 0                                         # zero the run length counter
    ls_failed = 0                          # no previous line search has failed
    f0 = f(X, *args)                          # get function value and gradient
    df0 = grad(X, *args)  
    fX = [f0]
    i = i + (length<0)                                         # count epochs?!
    s = -df0; d0 = -dot(s,s)    # initial search direction (steepest) and slope
    x3 = red/(1.0-d0)                             # initial step is red/(|s|+1)

    while i < abs(length):                                 # while not finished
        i = i + (length>0)                                 # count iterations?!

        X0 = X; F0 = f0; dF0 = df0              # make a copy of current values
        if length>0:
            M = MAX
        else: 
            M = min(MAX, -length-i)
        while 1:                      # keep extrapolating as long as necessary
            x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            success = 0
            while (not success) and (M > 0):
                try:
                    M = M - 1; i = i + (length<0)              # count epochs?!
                    f3 = f(X+x3*s, *args)
                    df3 = grad(X+x3*s, *args)
                    #if isnan(f3) or isinf(f3) or any(isnan(df3)+isinf(df3)):
                    if isnan(f3) or any(isnan(df3)+isinf(df3)):
                        print "an error in minimize error"
                        print "f3 = ", f3
                        print "df3 = ", df3
                        return
                    success = 1
                except:                    # catch any error which occured in f
                    x3 = (x2+x3)/2                       # bisect and try again
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3   # keep best values
            d3 = dot(df3,s)                                         # new slope
            if d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0:  
                                                   # are we done extrapolating?
                break
            x1 = x2; f1 = f2; d1 = d2                 # move point 2 to point 1
            x2 = x3; f2 = f3; d2 = d3                 # move point 3 to point 2
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)          # make cubic extrapolation
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
            Z = B+sqrt(complex(B*B-A*d1*(x2-x1)))
            if Z != 0.0:
                x3 = x1-d1*(x2-x1)**2/Z              # num. error possible, ok!
            else: 
                x3 = inf
            if (not isreal(x3)) or isnan(x3) or isinf(x3) or (x3 < 0): 
                                                       # num prob | wrong sign?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 > x2*EXT:           # new point beyond extrapolation limit?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 < x2+INT*(x2-x1):  # new point too close to previous point?
                x3 = x2+INT*(x2-x1)
            x3 = real(x3)

        while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:  
                                                           # keep interpolating
            if (d3 > 0) or (f3 > f0+x3*RHO*d0):            # choose subinterval
                x4 = x3; f4 = f3; d4 = d3             # move point 3 to point 4
            else:
                x2 = x3; f2 = f3; d2 = d3             # move point 3 to point 2
            if f4 > f0:           
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
                                                      # quadratic interpolation
            else:
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)           # cubic interpolation
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                if A != 0:
                    x3=x2+(sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
                                                     # num. error possible, ok!
                else:
                    x3 = inf
            if isnan(x3) or isinf(x3):
                x3 = (x2+x4)/2      # if we had a numerical problem then bisect
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))  
                                                       # don't accept too close
            f3 = f(X+x3*s, *args)
            df3 = grad(X+x3*s, *args)
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3              # keep best values
            M = M - 1; i = i + (length<0)                      # count epochs?!
            d3 = dot(df3,s)                                         # new slope

        if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:  # if line search succeeded
            X = X+x3*s; f0 = f3; fX.append(f0)               # update variables
            if verbose: print '%s %6i;  Value %4.6e\r' % (S, i, f0)
            s = (dot(df3,df3)-dot(df0,df3))/dot(df0,df0)*s - df3
                                                  # Polack-Ribiere CG direction
            df0 = df3                                        # swap derivatives
            d3 = d0; d0 = dot(df0,s)
            if d0 > 0:                             # new slope must be negative
                s = -df0; d0 = -dot(s,s)     # otherwise use steepest direction
            x3 = x3 * min(RATIO, d3/(d0-SMALL))     # slope ratio but max RATIO
            ls_failed = 0                       # this line search did not fail
        else:
            X = X0; f0 = F0; df0 = dF0              # restore best point so far
            if ls_failed or (i>abs(length)):# line search failed twice in a row
                break                    # or we ran out of time, so we give up
            s = -df0; d0 = -dot(s,s)                             # try steepest
            x3 = 1/(1-d0)                     
            ls_failed = 1                             # this line search failed
    if verbose: print "\n"
    return X, fX, i

def lognormal_logpdf( X, mu, sigma ):
  logpdf = -np.log(X) - 0.5*np.log(2*np.pi) - np.log(sigma) - 0.5*pow( (np.log(X)-mu)/sigma, 2 )
  return np.sum( logpdf )
  
def default_params():
  params = {}
  params["alpha"]           = 0.1
  params["beta"]            = 0.1
  params["theta_star"]      = 0.1
  params["N"]               = 500  # how many observations we draw per simulation
  params["q_stddev"]        = 0.5
  params["epsilon"]         = 0.1
  params["use_model"]       = False
  return params

class ExponentialProblem( BaseProblem ):
  # extract info about specific for this problem
  def load_params( self, params ):
    # prior parameters (Gamma distribution)
    self.alpha = params["alpha"]
    self.beta  = params["beta"]
    
    # proposal params (LogNormal)
    if params.has_key("q_stddev"):
      self.proposal_std    = params["q_stddev"]
    self.proposal_rand   = lognormal_rand
    self.proposal_logpdf = lognormal_logpdf
    #self.proposal_rand   = positive_normal_rand
    #self.proposal_logpdf = normal_logpdf
    
    # "true" parameter setting
    self.theta_star = params["theta_star"]
    
    # number of samples per simulation
    self.N         = params["N"]
    
    self.epsilon   = params["epsilon"]
    self.use_model = params["use_model"]
    
  # "create" problem or load observations  
  def initialize( self ):
    assert self.initialized is False, "Ensure we only call this once..."
    
    # generate observations and statistics
    np.random.seed(0)
    self.observations   = self.simulation_function( self.theta_star )
    # reproduce by setting seed(0) : 10.086749290513298
    self.obs_statistics = np.array([10.086749290513298]) #self.statistics_function( self.observations )
    self.obs_sum        = 10.086749290513298*self.N
    
    self.min_range           = 0.05
    self.max_range           = 0.15
    self.range               = (self.min_range,self.max_range)
    self.fine_bin_width      = 0.0001
    self.coarse_bin_width    = 0.001
    
    self.fine_theta_range    = np.arange( self.min_range, self.max_range+self.fine_bin_width, self.fine_bin_width )
    self.coarse_theta_range  = np.arange( self.min_range, self.max_range+self.coarse_bin_width, self.coarse_bin_width )
    
    self.nbins_coarse   = len(self.coarse_theta_range)
    self.nbins_fine     = len(self.fine_theta_range)
    self.log_posterior  = gamma_logprob( self.fine_theta_range, self.alpha+self.N, self.beta+self.obs_sum )
    self.posterior      = np.exp(self.log_posterior)
    self.posterior_mode = (self.N + self.alpha)/(self.obs_sum + self.beta)
    
    self.true_posterior_logpdf_func = gen_gamma_logpdf(self.alpha+self.N,self.beta+self.obs_sum)
    self.true_posterior_cdf_func    = gen_gamma_cdf(self.alpha+self.N,self.beta+self.obs_sum)
    
    self.posterior_bars_range = self.coarse_theta_range[:-1] + 0.5*self.coarse_bin_width
    self.posterior_cdf        = self.true_posterior_cdf_func( self.coarse_theta_range )
    self.posterior_bars       = (self.posterior_cdf[1:] - self.posterior_cdf[:-1])/self.coarse_bin_width
    self.posterior_cdf_bins   = self.posterior_cdf[1:] - self.posterior_cdf[:-1]
    # done initialization
    self.initialized = True
    
  def get_observations( self ):
    assert self.initialized, "Not initialized..."
    return self.observations
    
  def get_obs_statistics( self ):
    assert self.initialized, "Not initialized..."
    return self.obs_statistics
    
  def get_obs_groups( self ):
    assert self.initialized, "Not initialized..."

    params = {"response_type":"gaussian",
              "response_params":{"epsilon":self.epsilon }
             }
    g = ObservationGroup( np.array([0]), self.get_obs_statistics(), params )
    return [g]   
    
  # run simulation at parameter setting theta, return outputs
  def simulation_function( self, theta ):
    return np.random.exponential( 1.0/theta, self.N ) # 1/theta because of how python does exponential draws
    
  # pass outputs through statistics function, return statistics
  def statistics_function( self, outputs ):
    return np.array( [np.mean( outputs )] )
    
  # return size of statistics vector for this problem
  def get_nbr_statistics( self ):
    return 1
  
  # theta_rand
  def theta_prior_rand( self, N = 1 ):
    return np.random.gamma( self.alpha, 1.0/self.beta, N ) # 1/beta cause of how python implements
    
  # theta_rand
  def theta_prior_logpdf( self, theta ):
    return gamma_logprob( theta, self.alpha, self.beta ) # 1/beta cause of how python implements
      
  def theta_proposal_rand( self, theta ):
    #pdb.set_trace()
    log_q_theta = np.log(theta) + np.random.randn( len(theta) )*self.proposal_std
    return np.exp(log_q_theta)
    #return self.proposal_rand( theta, self.proposal_std )
    
  def theta_proposal_logpdf( self, to_theta, from_theta ):
    return lognormal_logpdf( to_theta, np.log( from_theta), self.proposal_std )
    #return self.proposal_logpdf( to_theta, from_theta, self.proposal_std )
    #return self.proposal_logpdf( to_theta, np.log(from_theta), self.proposal_std )
  
  def compute_errors_at_times( self, times, thetas, sims ):
    errs = []
    time_ids = []
    nbr_sims = []
    
    for time_id in times:
      if time_id <= len(thetas):
        errs.append( bin_errors_1d(self.coarse_theta_range, self.posterior_cdf_bins, thetas[:time_id]) )
        time_ids.append(time_id)
        nbr_sims.append(sims[:time_id].sum())
        
    errs = np.array(errs)
    time_ids = np.array(time_ids)
    nbr_sims = np.array(nbr_sims)
    
    return errs, nbr_sims, time_ids 
  
  def run_simulation_at_fixed_seeds( self, seed, grid=None ):
    ys = []
    thetas = []
      
    if grid is None:
      grid = self.coarse_theta_range
    for theta in grid:
      
      np.random.seed( seed )
      y = self.statistics_function(self.simulation_function(theta))
      ys.append(y)
      thetas.append(theta)
      
    return np.squeeze(np.array(thetas)), np.squeeze(np.array(ys))
  
  def mini_experiment( self, nbr_draws = 100 ):
    thetas = []
    ys     = []
    
    for seed in range(nbr_draws):
      x, y = self.run_simulation_at_fixed_seeds( seed )
      dif = self.obs_statistics - y
      iy = np.argmin( pow(dif,2) )
      ys.append( y[iy])
      thetas.append( x[iy])
      
    thetas = np.squeeze(np.array(thetas))
    ys = np.squeeze(np.array(ys))
    
    return thetas, ys 
  
  def view_simple( self, stats, thetas ):
    # plotting params
    nbins       = 20
    alpha       = 0.5
    label_size  = 8
    linewidth   = 3
    linecolor   = "r"
    
    # extract from states
    #thetas = states_object.get_thetas()[burnin:,:]
    #stats  = states_object.get_statistics()[burnin:,:]
    #nsims  = states_object.get_sim_calls()[burnin:]
    
    # plot sample distribution of thetas, add vertical line for true theta, theta_star
    f = pp.figure()
    sp = f.add_subplot(111)
    pp.plot( self.fine_theta_range, self.posterior, linecolor+"-", lw = 1)
    ax = pp.axis()
    pp.hist( thetas, self.nbins_coarse, range=self.range,normed = True, alpha = alpha )
    
    pp.fill_between( self.fine_theta_range, self.posterior, color="m", alpha=0.5)
    
    pp.plot( self.posterior_bars_range, self.posterior_bars, 'ro')
    pp.vlines( thetas.mean(), ax[2], ax[3], color="b", linewidths=linewidth)
    #pp.vlines( self.theta_star, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    pp.vlines( self.posterior_mode, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    
    pp.xlabel( "theta" )
    pp.ylabel( "P(theta)" )
    pp.axis([self.range[0],self.range[1],ax[2],ax[3]])
    set_label_fonsize( sp, label_size )
    pp.show()
          
  # take samples/staistics etc and "view" this particular problem
  def view_results( self, states_object, burnin = 1 ):
    # plotting params
    nbins       = 20
    alpha       = 0.5
    label_size  = 8
    linewidth   = 3
    linecolor   = "r"
    
    # extract from states
    thetas = states_object.get_thetas()[burnin:,:]
    stats  = states_object.get_statistics()[burnin:,:]
    nsims  = states_object.get_sim_calls()[burnin:]
    
    # plot sample distribution of thetas, add vertical line for true theta, theta_star
    f = pp.figure()
    sp = f.add_subplot(111)
    pp.plot( self.fine_theta_range, self.posterior, linecolor+"-", lw = 1)
    ax = pp.axis()
    pp.hist( thetas, self.nbins_coarse, range=self.range,normed = True, alpha = alpha )
    
    pp.fill_between( self.fine_theta_range, self.posterior, color="m", alpha=0.5)
    
    pp.plot( self.posterior_bars_range, self.posterior_bars, 'ro')
    pp.vlines( thetas.mean(), ax[2], ax[3], color="b", linewidths=linewidth)
    #pp.vlines( self.theta_star, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    pp.vlines( self.posterior_mode, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    
    pp.xlabel( "theta" )
    pp.ylabel( "P(theta)" )
    pp.axis([self.range[0],self.range[1],ax[2],ax[3]])
    set_label_fonsize( sp, label_size )
    
    total_sims = states_object.get_sim_calls().sum()
    all_sims = nsims.sum()
    at_burnin = total_sims-all_sims
    errs = []
    time_ids = []
    nbr_sims = []
    
    for time_id in [1,5,10,25,50,75,100,200,300,400,500,750,1000,1500,2000,3000,4000,5000,7500,10000,12500,15000,17500,20000,25000,30000,35000,40000,45000,50000]:
      if time_id <= len(thetas):
        errs.append( bin_errors_1d(self.coarse_theta_range, self.posterior_cdf_bins, thetas[:time_id]) )
        time_ids.append(time_id)
        nbr_sims.append(nsims[:time_id].sum()+at_burnin)
        
    errs = np.array(errs)
    time_ids = np.array(time_ids)
    nbr_sims = np.array(nbr_sims)
    
    f2 = pp.figure()
    sp1 = f2.add_subplot(1,3,1)
    pp.semilogx( time_ids, errs, "bo-", lw=2)
    pp.xlabel( "nbr samples")
    pp.ylabel( "err")
    pp.grid('on')
    sp2 = f2.add_subplot(1,3,2)
    pp.semilogx( nbr_sims, errs, "ro-", lw=2)
    pp.xlabel( "nbr sims")
    pp.ylabel( "err")
    pp.grid('on')
    sp3 = f2.add_subplot(1,3,3)
    pp.semilogx( time_ids, errs, "bo-", lw=2)
    pp.semilogx( nbr_sims, errs, "ro-", lw=2)
    pp.xlabel( "time")
    pp.ylabel( "err")
    pp.grid('on')
    pp.show()
    #pdb.set_trace()
    print "ERROR    ",bin_errors_1d( self.coarse_theta_range, self.posterior_cdf_bins, thetas )
    #print "ACC RATE ", states_object.acceptance_rate()
    print "SIM      ", total_sims
    # return handle to figure for further manipulation
    return f
    
if __name__ == "__main__":
  # view problem and rejection samples
  pp.rc('text', usetex=True)
  pp.rc('font', family='serif')
  #plt.xlabel(r'\textbf{time} (s)')
  
  epsilon = 2.0
  
  params = default_params()
  p = ExponentialProblem(params)
  p.initialize()
  thetas = np.load("./uai2014/saved/exponential/rejection_eps2p0_thetas.npy")[:,1]
  pp.close("all")
  # f1 = pp.figure(1)
  # sp1 = f1.add_subplot(111)
  # pp.plot( p.fine_theta_range, p.posterior, "k--", lw = 2)
  # 
  # pp.hist( thetas, p.nbins_coarse, range=p.range,normed = True, alpha = 0.25 )
  # ax = pp.axis()
  # pp.axis([p.range[0],p.range[1],ax[2],ax[3]])
  
  N = 2000
  good = []
  bad = []
  good_stats = []
  bad_stats = []
  for n in range(N):
    theta = p.theta_prior_rand()
    sim_outs = p.simulation_function(theta)
    stats = p.statistics_function(sim_outs)
    if np.abs( p.obs_statistics - stats )<= epsilon:
      good.append( theta )
      good_stats.append(stats)
    else:
      bad.append(theta)
      bad_stats.append(stats)
      
  good = np.squeeze(np.array(good))
  bad = np.squeeze(np.array(bad))
  good_stats = np.squeeze(np.array(good_stats))
  bad_stats = np.squeeze(np.array(bad_stats))
  
  figsize=(9,6)
  #dpi=600
  # f2 = pp.figure(2,figsize=figsize, dpi=dpi)
  # #f2=pp.figure(figsize=(3,3),dpi=300)
  # sp = f2.add_subplot(111)
  # 
  # pp.plot( bad, bad_stats, 'b.', ms=5,alpha=0.25)
  # pp.plot( good, good_stats, 'ro', ms=5, alpha = 0.5)
  # pp.vlines( p.posterior_mode, 0, 20 )
  # pp.hlines( p.obs_statistics, 0, 1.0 )
  # pp.hlines( p.obs_statistics+epsilon, 0, 1.0, linestyles="--", lw=2 )
  # pp.hlines( p.obs_statistics-epsilon, 0, 1.0, linestyles="--", lw=2 )
  # pp.axis( [0,1.0,0,20])
  # 
  # 
  # set_tick_fonsize( sp, 6 )
  # set_label_fonsize( sp, 8 )
  # 
  from mpl_toolkits.axes_grid1 import host_subplot
  import mpl_toolkits.axisartist as AA
  import matplotlib.pyplot as plt
  
  f=pp.figure(figsize=figsize)

  fs=16
  host = host_subplot(111, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  
  #host.plot( bad, bad_stats, 'r.', ms=5,alpha=0.25)
  host.plot( bad, bad_stats, 'bo', ms=7,alpha=0.25)
  host.plot( good, good_stats, 'ro', ms=15, alpha = 0.5)
  #host.vlines( p.posterior_mode, 0, 20 )
  host.hlines( p.obs_statistics, 0, 1.0 )
  host.hlines( p.obs_statistics+epsilon, 0, 1.0, linestyles="--", lw=4 )
  host.hlines( p.obs_statistics-epsilon, 0, 1.0, linestyles="--", lw=4 )
  #par2 = host.twinx()
  par1 = host.twinx()
  xx=np.linspace( 0.001, 0.2, 100 )
  
  #par1.plot( xx, np.exp( p.theta_prior_logpdf( xx) ), "b--", lw=1 )
  p2,=par1.plot( p.fine_theta_range, p.posterior, "b-", lw = 3)
  par1.hist( thetas, p.nbins_coarse, color="r",histtype="stepfilled",range=p.range,normed = True, alpha = 0.5 )
  
  par1.text(0.105,70,r'$\pi( \theta | y )$', fontsize=1.5*fs)
  par1.text(0.11,25,r'$\pi_{\epsilon}( \theta | y  )$', fontsize=1.5*fs)
  #pp.axis( [0,0.4,0,20])
  #par1.axis["right"].label.set_color("b")
  #par1.axis["right"].label.set_color(p2.get_color())
  host.set_xlim(0, 0.2)
  par1.set_xlim(0, 0.2)
  host.set_ylim(0, 20)
  par1.set_ylim(0, 250)
  host.axis["left"].set_label("x")
  host.axis["left"].label.set_rotation(90)
  
  #plt.xlabel(r'\textbf{time} (s)')
  #par1.axis["right"].set_label("p(theta|y)")
  par1.axis["right"].set_label(r'\textbf{$\pi( \theta | y )$}')
  host.axis["bottom"].set_label(r'\textbf{$\theta$}')
  host.axis["left"].label.set_fontsize(fs)
  host.axis["bottom"].label.set_fontsize(fs)
  host.axis["left"].major_ticklabels.set_fontsize(fs)
  host.axis["bottom"].major_ticklabels.set_fontsize(fs)
  par1.axis["right"].major_ticklabels.set_fontsize(fs)
  par1.axis["right"].label.set_fontsize(fs)
  par1.axis["right"].label.set_rotation(270)
  par1.axis["right"].label.set_color("b")
  
  host.set_title("Exponential problem", fontsize=20)
  f.savefig( "exponential_problem.eps", format="ps", dpi=600 ) #,bbox_inches="tight")
  #savefig( "test.png", format="png", dpi=300,bbox_inches="tight")
  
  # for tick in host.yaxis.get_major_ticks():
  #   tick.label.set_fontsize(6)
  # for tick in par1.yaxis.get_major_ticks():
  #   tick.label.set_fontsize(6)
  # 
  # set_tick_fonsize( host, 6 )
  # set_label_fonsize( host, 8 )
  # set_tick_fonsize( par1, 6 )
  # set_label_fonsize( par1, 8 )
  #     
  #fig, ax1 = pp.subplots(111)
  #ax2 = ax1.twinx()
  
  #sp2 = f2.add_subplot(111)
  # ax2.plot( p.fine_theta_range, p.posterior, "b-", lw = 2)
#   ax2.hist( thetas, p.nbins_coarse, range=p.range,normed = True, alpha = 0.25 )
#   ax = ax2.axis()
#   ax2.axis([p.range[0],p.range[1],0,300])
  pp.show()
  