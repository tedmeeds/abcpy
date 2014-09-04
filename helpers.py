

import scipy
from scipy import special
from scipy import stats
from scipy import integrate
from scipy.stats import chi2
from scipy.special import gammaln
from scipy.stats import norm as normmy
import pdb
import numpy as np
import scipy as sp
import pylab as pp
sqrt2 = np.sqrt( 2.0 )
sqrt_2pi = np.sqrt( 2.0 * np.pi )
log_sqrt_2pi = np.sqrt(sqrt_2pi)

a1 = 0.278393; loga1 = np.log( a1 )
a2 = 0.230389; loga2 = np.log( a2 )
a3 = 0.000972; loga3 = np.log( a3 )
a4 = 0.078188; loga4 = np.log( a4 )

def mvn_logpdf( X, mu, cov, invcov = None, logdet = None ):
  return log_pdf_full_mvn( X, mu, cov, invcov, logdet )
  
def mvn_diagonal_logpdf( X, mu, stddevs ):
  return log_pdf_diag_mvn( X, mu, stddevs )
  
def mvn_diagonal_logcdf( X, mu, stddevs ):
  logpdf = 0.0
  #pdb.set_trace()
  for x,mu,std in zip( X.T, mu, stddevs ):
    cdf = np.squeeze( normmy.cdf( (x-mu)/std ) )
    #print cdf, (x-mu)/std, x, mu, std
    #cdf = np.squeeze( normcdf( x, mu, std ) )
    if cdf ==0:
      logpdf += -np.inf
    else:
      logpdf += np.log( cdf )
  return logpdf
  
def mvn_diagonal_logcdfcomplement( X, mu, stddevs ):
  logpdf = 0.0
  #pdb.set_trace()
  for x,mu,std in zip( X.T, mu, stddevs ):
    cdf = np.squeeze( normmy.cdf( (x-mu)/std ) )
    #print cdf, (x-mu)/std, x, mu, std
    if cdf == 1:
      logpdf += -np.inf
    else:
      logpdf += np.log( 1.0-cdf )
  return logpdf
  
  #return log_pdf_diag_mvn( X, mu, stddevs )
  
def heavyside( X ):
  if X.__class__ == np.float64:
    if X < 0:
      return 0.0
    else:
      return 1.0
  else:
    Y = np.ones(X.shape)
    I = pp.find(X<0)
    Y[I] = 0
  
  return Y
  
def positive_normal_rand( mu, stddevs, N = 1 ):
  X = mu + stddevs*np.random.randn( N )
  
  I = pp.find( X <= 0 )
  i = 0
  while len(I) > 0:
    X[I] = mu + stddevs*np.random.randn( len(I) )
    J = pp.find( X[I] <= 0 )
    I = I[J]
    
  return X

def normal_logpdf( X, mu, stddevs ):
  return gaussian_logpdf( X, mu, stddevs)
    
def json_extract_from_list( param_list, key_name, key_value, value_names ):
  # go through param_list, looking for key_name == key_value
  # if found, extract values names from it
  
  bfound = False
  for params in param_list:
    #print params
    if params.has_key( key_name ):
      bfound = True
      if params[key_name] == key_value:
        extracts = []
        #print "starting extraction"
        for name in value_names:
          #print "\tchecking %s"%(name)
          if params.has_key( name ):
            extracts.append( params[name] )
            #print "\t\t",extracts
          else:
            #print "cannot find: %s"%(name)
            extracts.append( None )
            #print "\t\t",extracts
        #print "extracts", extracts
        return extracts
      else:
        pass #print "cannot find %s"%(key_value)
    else:
      pass #print "cannot find %s"%(key_name)
        
  if bfound is False:
    print "WARNING: json_extract_from_list could not find %s:%s"%(key_name, key_value)
  
  return [None for i in range(len(value_names))] 
  
# ** copied from spearmint **
def fast_distance( ls, x1, x2 = None ):
  if x2 is None:
      # Find distance with self for x1.

      # Rescale.
      xx1 = x1 / ls        
      xx2 = xx1

  else:
      # Rescale.
      xx1 = x1 / ls
      xx2 = x2 / ls
  
  r2 = np.maximum(-(np.dot(xx1, 2*xx2.T) 
                     - np.sum(xx1*xx1, axis=1)[:,np.newaxis]
                     - np.sum(xx2*xx2, axis=1)[:,np.newaxis].T), 0.0)

  return r2
  
# ** copied from spearmint **
def fast_grad_distance(ls, x1, x2=None):
    if x2 is None:
        x2 = x1
        
    # Rescale.
    x1 = x1 / ls
    x2 = x2 / ls
    
    N = x1.shape[0]
    M = x2.shape[0]
    D = x1.shape[1]
    gX = np.zeros((x1.shape[0],x2.shape[0],x1.shape[1]))

    code = \
    """
    for (int i=0; i<N; i++)
      for (int j=0; j<M; j++)
        for (int d=0; d<D; d++)
          gX(i,j,d) = (2/ls(d))*(x1(i,d) - x2(j,d));
    """
    try:
        scipy.weave.inline(code, ['x1','x2','gX','ls','M','N','D'], \
                       type_converters=scipy.weave.converters.blitz, \
                       compiler='gcc')
    except:
        # The C code weave above is 10x faster than this:
        for i in xrange(0,x1.shape[0]):
            gX[i,:,:] = 2*(x1[i,:] - x2[:,:])*(1/ls)

    return gX
    
    
def inv_wishart_rnd( nu, S ):
  return np.linalg.inv( wishart_rnd( nu, np.linalg.inv(S) ))
  
def wishart_rnd( nu, S, chol = None ):

  dim = S.shape[0]
  if chol is None:
    chol = np.linalg.cholesky(S)
  #nu = nu+dim - 1
  #nu = nu + 1 - np.arange(1,dim+1)
  a = np.zeros((dim,dim))
  
  for i in range(dim):
      for j in range(i+1):
          if i == j:
              a[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
          else:
              a[i,j]  = np.random.normal(0,1)
  return np.dot(chol, np.dot(a, np.dot(a.T, chol.T)))
  
  # cholesky = np.linalg.cholesky(S)
  # d = S.shape[0]
  # a = np.zeros((d,d),dtype=np.float32)
  # for r in xrange(d):
  #   if r!=0: a[r,:r] = np.random.normal(size=(r,))
  #   a[r,r] = np.sqrt(gamma_rnd(0.5*(nu-d+1),2.0))
  # return np.dot(np.dot(np.dot(cholesky,a),a.T),cholesky.T)
    
def campbell_method( X ):
  N,D = X.shape # D = v in campbell
  b1 = 2.0
  b2 = 1.25
  d0 = np.sqrt(D) + b1 / np.sqrt(2.0)
  
  w = np.ones(N) #/ np.float(N)
  x_bar = np.dot( w, X ) / w.sum()
  dif = X-x_bar
  #W = np.diag( w*w )
  S = np.zeros( (D,D))
  for i in range(N):
    S += pow(w[i],2)*np.outer( dif[i,:],dif[i,:] )
  S /= (np.sum(w*w)-1)
  
  #print "S1 ", S
  for t in range(3):
    invS = np.linalg.pinv(S)
    d = np.sqrt(np.sum(np.dot( dif, invS) * dif,1))
    #d = np.sqrt(np.diag(invS))
  
    x = np.ones(N)
    I = pp.find( d > d0 )
    w[I] = d0*np.exp(-0.5*((d[I]-d0)**2)/(b2**2))/d[I]
    #w /= d
  
    #pdb.set_trace()
    #pdb.set_trace()
    x_bar = np.dot( w, X ) / w.sum()
    dif = X-x_bar
    S = np.zeros( (D,D))
    for i in range(N):
      S += pow(w[i],2)*np.outer( dif[i,:],dif[i,:] )
    S /= (np.sum(w*w)-1)
    #invS = np.linalg.pinv(S)
  
  #print "S2 ", S
  #pdb.set_trace()
  return x_bar, S
  
def robust_estimators( samples ):
  Ns,Nr = samples.shape # number of dimensions, number of repetitions
  
  
  #S = samples.copy()
  mu_hat = samples.mean(1)
  S = samples - mu_hat.reshape((Ns,1))
  s_sq = S*S
  
  # (i)
  d_bar = pow(s_sq.sum(1)/Nr,0.5)
  D_bar = np.diag(d_bar)
  invD_bar = np.diag(1.0/d_bar)
  
  Q,R = np.linalg.qr( np.dot( S.T, invD_bar)/(Nr-1))
  invR = np.linalg.inv(R)
  invRT = np.linalg.inv(R.T)
  DRT = np.dot( D_bar, R.T)
  
  Sigma_bar    = np.dot( DRT, DRT.T ) 
  invSigma_bar = np.dot( np.dot(invD_bar,invR), np.dot(invRT,invD_bar) )
  
  # (ii)
  #dif = samples - mu_hat.reshape((Ns,1))
  m = np.sum( np.dot( invSigma_bar, S) * S, 0 )
  
  mp = np.zeros(100)
  for i in range(100):
    mp[i] = np.sum( np.dot( S[:,i].T,invSigma_bar )* S[:,i])
    
  #pdb.set_trace()
  # (iii)
  m0 = np.sqrt(Ns) + np.sqrt(2.0)
  w = np.ones( len(m) )
  I = pp.find( m > m0 )
  w[I] = np.exp( -(m0/m[I])*(m[I]-m0)**2)
  #w /= np.sum(w)
  
  # (iv)
  mu_hat = np.dot(samples,w)/np.sum(w)
  S = samples - mu_hat.reshape((Ns,1))
  
  # (v)
  s_sq = S*S
  d = pow(s_sq.sum(1)/Nr,0.5)
  D = np.diag(d)
  W = np.diag(w*2)
  invD = np.diag(1.0/d)
  
  Q,R = np.linalg.qr( np.dot(W,np.dot( S.T, invD))/np.sqrt((np.sum(w)-1)))
  DRT = np.dot( D, R.T)
  Sigma_hat = np.dot( DRT, np.dot( R, D ) ) 
  pdb.set_trace()
  return mu_hat, Sigma_hat
  
def gen_uniform_rand( left, right ):
  def uniform_rand( N = 1 ):
    return left + (right-left)*np.random.rand( N )
  return uniform_rand
  
def gen_uniform_logpdf(  left, right ):
  length = right-left
  def uniform_logpdf( theta ):
    if theta.__class__ == np.ndarray:
      lp = np.ones( theta.shape ) / length
      I = pp.find( theta < left )
      J = pp.find( theta > right )
      lp[I] = -np.inf
      lp[J] = -np.inf
      return lp
    else:
      if theta < left or theta > right:
        return -np.inf
      else:
        return 1.0 / length
  return uniform_logpdf

def gen_uniform_cdf(  left, right ):
  length = right-left
  def uniform_cdf( theta ):
    return (theta-left)/length
  return uniform_cdf
  
def gen_uniform_icdf(  left, right ):
  length = right-left
  def uniform_icdf( u ):
    return length*u + left
  return uniform_icdf 
    
def gen_beta_rand( alpha, beta ):
  def beta_rand( N = 1 ):
    return np.random.beta( alpha, beta, N )
  return beta_rand
  
def gen_beta_logpdf( alpha, beta ):
  def beta_logpdf( theta ):
    return beta_logprob( theta, alpha, beta )
  return beta_logpdf
  
def gen_beta_cdf( alpha, beta ):
  g = stats.beta(alpha,beta)
  def beta_cdf( theta ):
    return g.cdf(theta)
  return beta_cdf
  
def gen_beta_icdf( alpha, beta ):
  g = stats.beta(alpha,beta)
  def beta_icdf( u ):
    return g.ppf(u)
  return beta_icdf

def gen_gamma_rand( alpha, beta ):
  def gamma_rand( N = 1 ):
    return gamma_rnd( alpha, beta, N )
  return gamma_rand
  
def gen_gamma_logpdf( alpha, beta ):
  def gamma_logpdf( theta ):
    return gamma_logprob( theta, alpha, beta  )
  return gamma_logpdf
  
def gen_gamma_cdf( alpha, beta ):
  g = stats.gamma(alpha,0,1.0/beta)
  def gamma_cdf( theta ):
    return g.cdf(theta)
  return gamma_cdf
  
def gen_gamma_icdf( alpha, beta ):
  g = stats.gamma(alpha,0,1.0/beta)
  def gamma_icdf( u ):
    return g.ppf(u)
  return gamma_icdf
    
def gen_gaussian_rand( mu, sigma ):
  def gaussian_rand( N = 1 ):
    return mu + sigma*np.random.randn( mu, sigma, N )
  return gaussian_rand
  
def gen_gaussian_logpdf( mu, sigma ):
  def gaussian_logpdf( theta ):
    return gaussian_logpdf( theta, mu, sigma  )
  return gaussian_logpdf
  
def gen_gaussian_cdf( mu, sigma ):
  def gaussian_cdf( theta ):
    return normcdf( theta, mu, sigma )
  return gaussian_cdf
  
def gen_gaussian_icdf( mu, sigma ):
  g = stats.norm( mu, sigma )
  def gaussian_icdf( u ):
    return g.ppf(u)
  return gaussian_icdf
    
    
def gen_all_uniform( alpha, beta ):
  return gen_uniform_rand(alpha,beta),\
         gen_uniform_logpdf(alpha,beta),\
         gen_uniform_cdf(alpha,beta),\
         gen_uniform_icdf(alpha,beta)
             
def gen_all_gamma( alpha, beta ):
  return gen_gamma_rand(alpha,beta),\
         gen_gamma_logpdf(alpha,beta),\
         gen_gamma_cdf(alpha,beta),\
         gen_gamma_icdf(alpha,beta)

def gen_all_beta( alpha, beta ):
  return gen_beta_rand(alpha,beta),\
         gen_beta_logpdf(alpha,beta),\
         gen_beta_cdf(alpha,beta),\
         gen_beta_icdf(alpha,beta)

def gen_all_gaussian( mu, sigma ):
  return gen_gaussian_rand(mu, sigma),\
         gen_gaussian_logpdf(mu, sigma),\
         gen_gaussian_cdf(mu, sigma),\
         gen_gaussian_icdf(mu, sigma)
                  
def test_rank_one_schur( N, D, Astar=None ):
  
  if Astar is None:
    X = np.random.randn( N, D )
    Astar = np.dot( X.T, X )
  
  
  A = Astar[:-1,:][:,:-1]
  b = Astar[:-1,-1]
  a = Astar[-1,-1]
  
  Astar_stack = rank_one_stack(A,b,a)
  
  Ainv = np.linalg.inv(A)
  Astarinv = np.linalg.inv(Astar)
  
  Astarinv_schur, schurcompInv, schurcompInvDirect = rank_one_schur_update( A, Ainv, b, a )
  
  print "schur comp inv"
  print schurcompInv-schurcompInvDirect
  print "Astarinv"
  print Astarinv-Astarinv_schur
  return A, Astar, Astar_stack, Ainv, Astarinv, Astarinv_schur, schurcompInv, schurcompInvDirect
  
def rank_one_stack( A, b, a ):
  #              [ A   b]
  # create A* = [b^t a ]
  #
  top_row = np.hstack( (A,b))
  bot_row = np.hstack( (b.T,a))
  
  Astar = np.vstack( (top_row, bot_row ))
  
  return Astar

def rank_one_schur_update_old( A, invA, b, a ):
  #                      [ A   b]
  # find inverse of A* = [b^t a ]
  #
  
  # 1) find inverse of schur complement (A+b b^t / a) ^-1
  Ainvb = np.dot( invA, b )
  f = 1.0 / (a + np.dot( b.T, Ainvb) )
  schurcompInv = invA - np.dot( Ainvb, Ainvb.T )/f
  schurcompInvDirect = np.linalg.inv( A-np.dot(b,b.T)/a)
  
  schurcompInv = schurcompInvDirect
  
  # 2) find new inv(A*)
  Sinvb = np.dot( schurcompInv,b)
  top_right = -Sinvb/a
  top_row = np.vstack( (schurcompInv, top_right) ).T
  bottom_right = 1.0/a + np.dot( b.T, Sinvb )/(a*a)
  bot_row = np.hstack( (top_right.T, bottom_right )) 
  
  Astarinv = np.vstack( (top_row,bot_row))
  
  return Astarinv, schurcompInv, schurcompInvDirect

def rank_one_schur_update( A, invA, b, a ):
  #                      [ A   b]
  # find inverse of A* = [b^t a ]
  #
  
  # 1) 
  Ainvb = np.dot( invA, b )
  f = a -  np.dot( b.T, Ainvb)
  
  # 2) find new inv(A*)
  #Sinvb = np.dot( schurcompInv,b)
  top_left = invA - np.dot( Ainvb, Ainvb.T )/f
  top_right = -Ainvb/f
  bot_right = 1.0/f
  
  Astarinv = rank_one_stack(top_left, top_right, bot_right )
  
  return Astarinv
    
def pdf_difference_in_scaled_nc_squared( z, v_left, v_right, mu_x, mu_y, s_x, s_y ):
  # assume z = y-z
  #normconstant = pow( b_x, a_x )*pow( b_y, a_y )/(special.gamma(a_y)*special.gamma(a_x))
  v_y = pow(s_y,2)
  v_x = pow(s_x,2)
  m_x = pow(mu_x,2)
  m_y = pow(mu_y,2)
  
  # only do positive
  #z = np.abs(z)
  
  p = np.exp( -0.5*(m_y/v_y+m_x/v_x) )/(2.0*np.pi*s_x*s_y)*np.ones(len(z))
  
  for i,zi in zip(range(len(z)),z):
    if zi >=0:
      f_i = gen_scaled_nc_diff_integral( zi, mu_x, mu_y, s_x, s_y )
      
      p[i] *= np.exp( -0.5*zi/v_y)*integrate.quad( f_i, v_left, v_right )[0]
    else:
      f_i = gen_scaled_nc_diff_integral( -zi, mu_y, mu_x, s_y, s_x )
  
      p[i] *= np.exp( 0.5*zi/v_x)*integrate.quad( f_i, v_left, v_right )[0]
    
  return p
  
def gen_scaled_nc_diff_integral( z, mu_x, mu_y, s_x, s_y ):
  def hard_integral( x ):
    v_y = pow(s_y,2)
    v_x = pow(s_x,2)
    m_x = pow(mu_x,2)
    m_y = pow(mu_y,2)
    
    f1 = pow( x, -0.5)
    f2 = pow( x+z, -0.5)
    f3 = np.exp( -0.5*x*(1.0/v_y+1.0/v_x) )
    f4 = np.cosh( np.sqrt(m_y*(x+z))/v_y )
    f5 = np.cosh( np.sqrt(m_x*x)/v_x )
    
    return f1*f2*f3*f4*f5 
  return hard_integral
  
def pdf_difference_in_nc_squared( z, v_left, v_right, nc_x, nc_y ):
  # assume z = y-z
  #normconstant = pow( b_x, a_x )*pow( b_y, a_y )/(special.gamma(a_y)*special.gamma(a_x))
  
  # only do positive
  z = np.abs(z)
  
  p = np.exp( -0.5*(z+nc_x+nc_y) )/(2.0*np.pi)
  
  for i,zi in zip(range(len(z)),z):
    f_i = gen_nc_diff_integral( zi, nc_x, nc_y )
  
    p[i] *= integrate.quad( f_i, v_left, v_right )[0]
    
  return p
  
def gen_nc_diff_integral( z, nc_x, nc_y ):
  def hard_integral( x ):
    f1 = pow( x, -0.5)
    f2 = pow( x+z, -0.5)
    f3 = np.exp( -x )
    f4 = np.cosh( np.sqrt((x+z)*nc_y) )
    f5 = np.cosh( np.sqrt(x*nc_x ) )
    
    return f1*f2*f3*f4*f5 
  return hard_integral
  
def pdf_difference_in_gammas( z, v_left, v_right, a_y, a_x, b_y, b_x ):
  # assume z = y-z
  normconstant = pow( b_x, a_x )*pow( b_y, a_y )/(special.gamma(a_y)*special.gamma(a_x))
  
  # only do positive
  z = np.abs(z)
  
  p = normconstant*np.exp( -z*b_y )
  
  for i,zi in zip(range(len(z)),z):
    f_i = gen_gamma_diff_integral( zi, a_y, a_x, b_y, b_x )
  
    p[i] *= integrate.quad( f_i, v_left, v_right )[0]
    
  return p
  
def gen_gamma_diff_integral( z, a_y, a_x, b_y, b_x ):
  def hard_integral( x ):
    f1 = pow( x, a_x-1.0)
    f2 = pow( x+z, a_y-1.0)
    f3 = np.exp( -x*(b_x+b_y))
    
    return f1*f2*f3 
  return hard_integral


def pdf_using_marcum(x, v_left, v_right, mu_p, mu_t, s_p, s_t, s_mu_p, s_mu_t):
  v_p = pow(s_p,2)
  v_t = pow(s_t,2)
  v_mu_p = pow(s_mu_p,2)
  v_mu_t = pow(s_mu_t,2)
  
  a = 1.0/(v_mu_p)
  b = 1.0/(v_mu_t)
  e = (mu_p**2) / v_mu_p
  f = (mu_t**2) / v_mu_t
  
  print "mu_p   = %0.1f"%(mu_p)
  print "mu_t   = %0.1f"%(mu_t)
  print "s_p    = %0.3f"%(s_p)
  print "s_t    = %0.3f"%(s_t)
  print "s_mu_p = %0.3f"%(s_mu_p)
  print "s_mu_t = %0.3f"%(s_mu_t)
  
  aa = sqrt2*np.abs(mu_p)/v_mu_p
  bb = sqrt2*np.abs(mu_t)/v_mu_t
  
  norm_const = 1.0/pow(a*b,2.0)
  
  print norm_const, a,b,e,f,aa,bb
  
  if x.__class__ == np.ndarray:
    n = len(x)
    integral = np.zeros(n)
    for i in range(n):
      if x[i] > 0:
        c=b
        hi = gen_marcum_integral( x[i], mu_p, mu_t, v_mu_p, v_mu_t )
        integral[i] = np.exp(nc_simplified_from_chi_log_prob( x[i], mu_p**2, 1.0 ))
        #integral[i] = norm_const*np.exp( -np.abs(x[i])*c )*integrate.quad( hi, v_left, v_right )[0]
      else:
        c=a
        hi = gen_marcum_integral( -x[i], mu_t, mu_p, v_mu_t, v_mu_p )
        #integral[i] = norm_const*integrate.quad( hi, v_left, v_right )[0]
        integral[i] = np.exp(nc_simplified_from_chi_log_prob( -x[i], mu_t, 1.0 ))
      
  else:
    assert False, "incomplete"
    hi = gen_hard_integral( x, a, b )
    integral = integrate.quad( hi, max(-x,v_left), v_right )[0]
  
  return integral
  
def gen_marcum_integral( x, mu_p, mu_t, v_mu_p, v_mu_t ):
  def hard_integral( u ):
    f1 = marcum_q( 1.0, np.sqrt(mu_t*mu_t/v_mu_t), np.sqrt(u+x) )
    lp = nc_simplified_from_chi_log_prob( u, mu_p, 1.0 )
    #lp = noncentral_chisquare_standard_log_pdf( u, 1.0, mu_p**2)
    #nc_simplified_from_chi_log_prob( x_range_up, self.mu_p, self.s_mu_p )
    f2 = np.exp(lp)
    return f1*f2
  return hard_integral
  
def pdf_hard_at_x( x, v_left, v_right, mu_p, mu_t, s_p, s_t, s_mu_p, s_mu_t ):
  v_p = pow(s_p,2)
  v_t = pow(s_t,2)
  v_mu_p = pow(s_mu_p,2)
  v_mu_t = pow(s_mu_t,2)
  
  a = 1.0/(v_mu_p)
  b = 1.0/(v_mu_t)
  e = (mu_p**2) / v_mu_p
  f = (mu_t**2) / v_mu_t
  
  print "mu_p   = %0.1f"%(mu_p)
  print "mu_t   = %0.1f"%(mu_t)
  print "s_p    = %0.3f"%(s_p)
  print "s_t    = %0.3f"%(s_t)
  print "s_mu_p = %0.3f"%(s_mu_p)
  print "s_mu_t = %0.3f"%(s_mu_t)
  
  aa = sqrt2*np.abs(mu_p)/v_mu_p
  bb = sqrt2*np.abs(mu_t)/v_mu_t
  
  norm_const = np.exp(-0.5*(e+f)) / (2*s_mu_t*s_mu_p*np.pi)
  
  print norm_const, a,b,e,f,aa,bb
  
  if x.__class__ == np.ndarray:
    n = len(x)
    integral = np.zeros(n)
    for i in range(n):
      if x[i] > 0:
        c=b
        #bb = (a/s_p)*np.sqrt(2*pow(mu_p,2))
        #aa = (b/s_t)*np.sqrt(2*pow(mu_t,2))
        hi = gen_hard_integral( x[i], a + b, bb,aa )
        #integral[i] = norm_const*np.exp( -np.abs(x[i])*c )*integrate.quad( hi, v_left, v_right )[0]
      else:
        c=a
        #aa = (a/s_p)*np.sqrt(2*pow(mu_p,2))
        #bb = (b/s_t)*np.sqrt(2*pow(mu_t,2))
      
        hi = gen_hard_integral( x[i], a + b, aa, bb )
      integral[i] = norm_const*np.exp( -np.abs(0.5*x[i])*c )*integrate.quad( hi, v_left, v_right )[0]
      
  else:
    assert False, "incomplete"
    hi = gen_hard_integral( x, a, b )
    integral = integrate.quad( hi, max(-x,v_left), v_right )[0]
  
  return integral
  
  
def gen_hard_integral( x, a_plus_b, aa, bb ):
  def hard_integral( v ):
    f1 = 1.0/np.sqrt(v)
    f2 = 1.0/np.sqrt(v+np.abs(x))
    f3 = np.exp( -0.5*v*a_plus_b )
    f4 = np.cosh(aa*np.sqrt(v+np.abs(x)))
    f5 = np.cosh(bb*np.sqrt(np.abs(x)))
    
    f3 = np.exp( aa*np.sqrt(v+np.abs(x)) + bb*np.sqrt(np.abs(x))  -0.5*v*a_plus_b )/4.0
    return f1*f2*f3 #*f4*f5
  return hard_integral
  
  

class likelihood_ratio_acceptance(object):
  def __init__( self, mu_p, sigma_p, sigma_mu_p, mu_t, sigma_t, sigma_mu_t, N = 100000 ):
    self.mu_p    = mu_p
    self.s_p     = sigma_p
    self.s_mu_p  = sigma_mu_p
    self.mu_t    = mu_t
    self.s_t     = sigma_t
    self.s_mu_t  = sigma_mu_t
    
    self.v_p = pow(self.s_p,2)
    self.v_t = pow(self.s_t,2)
    
    self.v_mu_p = pow( self.s_mu_p, 2 )
    self.v_mu_t = pow( self.s_mu_t, 2 )
    
    print "--> mu_p   = %0.1f"%(self.mu_p)
    print "--> s_p    = %0.3f"%(self.s_p)
    print "--> s_mu_p = %0.3f"%(self.s_mu_p )
    print "--> mu_t   = %0.1f"%(self.mu_t)
    print "--> s_t    = %0.3f"%(self.s_t)
    print "--> s_mu_t = %0.3f"%(self.s_mu_t )
    
    self.simulate( N )
    
  def simulate( self, N ):
    # the means sampled from their prior
    mups = self.mu_p + self.s_mu_p*np.random.randn(N)
    muts = self.mu_t + self.s_mu_t*np.random.randn(N)
    
    x_range_mups = np.linspace( self.mu_p-3*self.s_mu_p, self.mu_p+3*self.s_mu_p, 100 )
    x_range_muts = np.linspace( self.mu_t-3*self.s_mu_t, self.mu_t+3*self.s_mu_t, 100 )
    
    lp_mups = gaussian_logpdf( x_range_mups, self.mu_p, self.s_mu_p )
    lp_muts = gaussian_logpdf( x_range_muts, self.mu_t, self.s_mu_t )
    
    # the squared means
    up = mups**2
    ut = muts**2
    
    mean_up = self.v_mu_p + pow(self.mu_p,2)
    var_up  = 2*self.v_mu_p*mean_up
    mean_ut = self.v_mu_t + pow(self.mu_t,2)
    var_ut  = 2*self.v_mu_t*mean_ut
    
    x_range_up = np.linspace( max(0,mean_up-3*pow(var_up,0.5) ), mean_up+5*pow(var_up,0.5), 100 )
    x_range_ut = np.linspace( max(0,mean_ut-3*pow(var_ut,0.5) ), mean_ut+5*pow(var_ut,0.5), 100 )
    
    lp_up = nc_simplified_from_chi_log_prob( x_range_up, self.mu_p, self.s_mu_p )
    lp_ut = nc_simplified_from_chi_log_prob( x_range_ut, self.mu_t, self.s_mu_t )
    
    # the scaled squares
    fup = up / (2.0*self.v_p )
    fut = ut / (2.0*self.v_t )
    
    mean_fup = mean_up/(2.0*self.v_p)
    var_fup  = var_up/pow(2.0*self.v_p,2)
    mean_fut = mean_ut/(2.0*self.v_t)
    var_fut  = var_ut/pow(2.0*self.v_t,2)
    
    x_range_fup = np.linspace( max(0,mean_fup-3*pow(var_fup,0.5) ), mean_fup+5*pow(var_fup,0.5), 100 )
    x_range_fut = np.linspace( max(0,mean_fut-3*pow(var_fut,0.5) ), mean_fut+5*pow(var_fut,0.5), 100 )
    
    lp_fup = nc_simplified_from_chi_scaled_log_prob( x_range_fup, self.mu_p, self.s_mu_p, self.s_p )
    lp_fut = nc_simplified_from_chi_scaled_log_prob( x_range_fut, self.mu_t, self.s_mu_t, self.s_t )
    
    # the difference (within the exponential of likelihood ratio)
    z = ut - up
    
    mean_z = z.mean()
    var_z  = z.var()
    
    pos_z_range = np.linspace( 0.01,5.0, 200 )
    neg_z_range = np.linspace( -5.0,-0.01, 200 )
    
    a = self.v_p / self.v_mu_p
    b = self.v_t / self.v_mu_t
    e = (self.mu_p**2) / self.v_p
    f = (self.mu_t**2 )/ self.v_t
    
    #print a,b,e,f
    #p_z = pdf_hard_at_x( z_range, 0, 10.0, a, b, e, f, self.s_p, self.s_t, self.mu_p, self.mu_t )
    p_pos_z = pdf_hard_at_x( pos_z_range, 0, 100.0, self.mu_p, self.mu_t, self.s_p, self.s_t, self.s_mu_p, self.s_mu_t )
    #p_neg_z = pdf_hard_at_x( neg_z_range, 0, 100.0, self.mu_p, self.mu_t, self.s_p, self.s_t, self.s_mu_p, self.s_mu_t )
    p_neg_z = pdf_using_marcum( neg_z_range, 0, 100.0, self.mu_p, self.mu_t, self.s_p, self.s_t, self.s_mu_p, self.s_mu_t )
    
    #p_pos_z = pdf_using_marcum( pos_z_range, 0, 100.0, self.mu_p, self.mu_t, self.s_p, self.s_t, self.s_mu_p, self.s_mu_t )
    # pos_x_range = np.linspace( 0.001, 0.5, 100 )
#     neg_x_range = np.linspace( -1.0, -0.001, 100 )
#     pos_x = pdf_hard_at_x( pos_x_range, 0.0,20,a,b)
#     neg_x = pdf_hard_at_x( neg_x_range, 0.0,20,a,b)
    
    pp.figure()
    #ax1 = pp.subplot2grid( (3,3), (0,0), colspan=3)
    pp.subplot( 3, 2 , 1 )
    pp.hist( mups, 50, normed=True, alpha=0.5 )
    pp.plot(x_range_mups, np.exp(lp_mups), 'r-', lw=2 )
    pp.xlabel( "mu_p")
    pp.subplot( 3, 2 , 2 )
    pp.hist( muts, 50, normed=True, alpha=0.5 )
    pp.plot(x_range_muts, np.exp(lp_muts), 'r-', lw=2 )
    pp.xlabel( "mu_t")
    pp.subplot( 3, 2 , 3 )
    pp.hist( up, 50, normed=True, alpha=0.5 )
    pp.plot(x_range_up, np.exp(lp_up), 'r-', lw=2 )
    pp.xlabel( "mu_p^2")
    pp.subplot( 3, 2 , 4 )
    pp.hist( ut, 50, normed=True, alpha=0.5 )
    pp.plot(x_range_ut, np.exp(lp_ut), 'r-', lw=2 )
    pp.xlabel( "mu_t^2")
    pp.subplot( 3, 2 , 5 )
    pp.hist( fup, 50, normed=True, alpha=0.5 )
    pp.plot(x_range_fup, np.exp(lp_fup), 'r-', lw=2 )
    pp.xlabel( "mu_p^2 / 2*v_p")
    pp.subplot( 3, 2 , 6 )
    pp.hist( fut, 50, normed=True, alpha=0.5 )
    pp.plot(x_range_fut, np.exp(lp_fut), 'r-', lw=2 )
    pp.xlabel( "mu_t^2 / 2*v_t")
    
    pp.figure()
    pp.subplot(3,1,1)
    pp.hist(z, 200, normed=True, alpha=0.5)
    #pp.plot(neg_z_range, p_neg_z, 'r-', lw=2)
    #pp.plot(pos_z_range, p_pos_z, 'c-', lw=2)
    pp.subplot(3,1,2)
    pp.hist(z, 200, normed=True, alpha=0.5)
    #pp.plot(neg_z_range, p_neg_z, 'r-', lw=2)
    pp.plot(neg_z_range, p_neg_z, 'r-', lw=2)
    pp.plot(pos_z_range, p_pos_z, 'c-', lw=2)
    #I = pp.find(z_range>0)
    #pp.plot(z_range[I], p_z[I], 'r-', lw=2)
    #pp.plot(neg_z_range, p_neg_z, 'c-', lw=2)
    pp.subplot(3,1,3)
    #I = pp.find(z_range<0)
    #pp.plot(z_range[I], p_z[I], 'r-', lw=2)
    pp.plot(neg_z_range, p_neg_z, 'r-', lw=2)
    pp.plot(pos_z_range, p_pos_z, 'c-', lw=2)
    # constant
    # c = self.s_t/self.s_p
    # logc = np.log(c)
    # 
    # zc = z + logc
    # 
    # I_acc = pp.find(zc >= 0)
    # I_check = pp.find(zc < 0)
    # 
    # exp_zc = np.exp(zc)
    # 
    # a = np.ones( N )
    # a[I_check] = exp_zc[I_check]
    # 
    # mean_a = a.mean()
    
  
def marcum_q( m, a, b ):
  s = np.exp( -0.5*(pow(a,2)+pow(b,2)))
  
  max_k = 3
  sm = 0
  for ki in range(max_k):
    k = 1.0 - m + ki
    sm += pow( a/b, k)*special.iv(k,a*b)
    #print "\t",sm
    
  #print s,1./s,sm
  return s*sm
  
def nc_simplified_from_chi_log_prob2( x, sigma, mean ):
  # sigma and mean from prior on y = ~ N(mean,sigma^2), x = y^2
  m = mean**2
  v = sigma**2
  lp = -0.5*np.log(2*np.pi) - np.log(sigma) - 0.5*(x + m)/v - 0.5*np.log(x) + np.sqrt(m) * np.sqrt(x)/v 
  
  return lp
  
def nc_simplified_from_chi_scaled_log_prob( x, mean, sigma, sigma2 ):
  # sigma and mean from prior on y = ~ N(mean,sigma^2), x = y^2
  m = mean**2
  v = sigma**2
  v2 = sigma2**2
  scale_in_var = v
  lp = -0.5*np.log(np.pi) + np.log(sigma2/sigma)  \
      - v2*x/v + -0.5*m/v - 0.5*np.log(x) + np.log(np.cosh( np.sqrt(m*x*2)*sigma2/v ))

  return lp
  
def nc_simplified_from_chi_log_prob( x, mean, sigma ):
  # sigma and mean from prior on y = ~ N(mean,sigma^2), x = y^2
  m = mean**2
  v = sigma**2
  scale_in_var = v
  lp = -0.5*np.log(2*np.pi) - np.log(sigma) - 0.5*(x + m)/v - 0.5*np.log(x) + np.log(np.cosh( np.sqrt(m) * np.sqrt(x)/v ))

  return lp


def noncentral_chisquare_standard_log_pdf( x, ddof, noncentrality):
  lp = np.log(0.5)-0.5*(x+noncentrality) + (ddof/4.0 - 0.5)*np.log( x / noncentrality) + 0.5*np.log(2.0/(np.pi*np.sqrt(x*noncentrality))) + np.log(np.cosh(np.sqrt(x*noncentrality)))#+ np.log(special.iv(ddof/2.0-1,np.sqrt(x*noncentrality))) #+ 0.5*np.log(2.0/(np.pi*np.sqrt(x*noncentrality))) + np.log(np.sinh(np.sqrt(x*noncentrality))) #np.log(special.iv(ddof-1,np.sqrt(x*noncentrality)))
  return lp
  
def scaled_noncentral_chi_squared_log_pdf( x, ddof, noncentrality, scale_in_var ):
  #lp = noncentral_chisquare_log_pdf( x/scale_in_var, ddof, noncentrality/scale_in_var ) - np.log(scale_in_var)
  
  lp = noncentral_chisquare_simplified_log_pdf( x/scale_in_var, noncentrality/scale_in_var)- np.log(scale_in_var)
  return lp
  
def noncentral_chisquare_simplified_log_pdf( x, noncentrality ):
  lp = -0.5*np.log(2*np.pi) -0.5*(x+noncentrality) - 0.5*np.log(x) + np.log( np.cosh( np.sqrt(x*noncentrality) ))
  return lp
  
def noncentral_chisquare_log_pdf( x, ddof, noncentrality ):
  assert ddof > 0, "ddof must be > 0"
  assert noncentrality > 0, "noncentrality must be > 0"
  
  # this version lacks log modified bessel of first kind
  # lp = -np.log(2) - 0.5*(x+noncentrality) + (ddof/4.0 - 0.5)*np.log( x / noncentrality )
  
  lp = - 0.5*(x+noncentrality) - 0.5*ddof*np.log(2.0) + (0.5*ddof-1)*np.log(x) - special.gammaln( 0.5*ddof )
  lp += np.log( special.hyp0f1( ddof/2.0, noncentrality*x/4.0 ) )
  return lp

def log_prob_exp_of_squared( x, s = 1.0 ):
  # assume y ~ N(0,1)
  # x = exp( y^2 )
  # return p(x)
  return gamma_logprob( np.log(x), 0.5, 0.5 ) - np.log(x) + np.log(s)

def log_stdnormcdf( x ):
  return logsumexp( np.array([ np.log(0.5),np.log(0.5)+log_erf( x/np.sqrt(2) )]))
  
def stdnormcdf_appr( x ):
  return np.exp( log_stdnormcdf( x ) )
  
def log_erf( x ):
  # 1 - erfc(x)
  return np.log( 1 - erfc_appr(x) )
  
def erfc_appr( x ):
  if x < 0:
    log_erf_x = log_erfc( -x )
    return 2-np.exp( log_erf_x )
  else:
    log_erf_x = log_erfc( x )
    return np.exp( log_erf_x )
    
def log_erfc( x ):
  if x < 0:
    log_erf = log_erfc( -x )
    return np.log( 2-np.exp( log_erf ) )
  elif x == 0:
    return 0*x
    
  #assert x >= 0, "log_erf_f assumes positive x"
  logx = np.log(x)
  # erf(x) = 1 - erfc(x)
  # erfc(x) = 1 / (1+a1*x +a2*x^2 + a3*x^3 + a4*x^4)^4
  denom = 4*logsumexp( np.array([0,loga1+logx,loga2+2*logx,loga3+3*logx,loga4+4*logx]))
  
  return - denom
    
  
def gamma_rnd( alpha, beta, N=1 ):
  return np.random.gamma( alpha, 1.0/beta, N )
  
def invgamma_rnd( alpha, beta, N = 1):
  #stats.invgamma.rvs( 3.0, scale=1.0/2.0,size=1000 )
  return stats.invgamma.rvs( alpha, scale = beta, size=N)

def invgamma_logprob( x, alpha, beta):
  return stats.invgamma.logpdf( x, alpha, scale = beta)

def invgamma_logprob_gradient_x(x, alpha, beta):   
  #print " *************************************************************************** "
  #print " WARNING: invgamma_logprob_gradient called but it really is invgamma_logprob " 
  #print " *************************************************************************** "
  #return stats.invgamma.logpdf( x, alpha, scale = beta)
  return -(alpha+1)/x + beta/pow(x,2)

def invgamma_logprob_gradient_free_x( free_x, alpha, beta):   
  #print " *************************************************************************** "
  #print " WARNING: invgamma_logprob_gradient called but it really is invgamma_logprob " 
  #print " *************************************************************************** "
  #return stats.invgamma.logpdf( x, alpha, scale = beta)
  x = np.exp(free_x )
  # this without transform
  #return -(alpha+1)/x + beta/pow(x,2)
  
  # this with
  return -(alpha+1) + beta/x
    
def logsumexp(x,dim=0):
    """Compute log(sum(exp(x))) in numerically stable way."""
    #xmax = x.max()
    #return xmax + log(exp(x-xmax).sum())
    if dim==0:
        xmax = x.max(0)
        return xmax + np.log(np.exp(x-xmax).sum(0))
    elif dim==1:
        xmax = x.max(1)
        return xmax + np.log(np.exp(x-xmax[:,np.newaxis]).sum(1))
    else: 
        raise 'dim ' + str(dim) + 'not supported'
        
def gamma_logprob( x, alpha, beta ):
  if all(x>0):
    if beta > 0:
      return alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x) - beta*x
    else:
      assert False, "Beta is zero"
  else:
    if x.__class__ == np.array:
      I = pp.find( x > 0.0 )
      lp = -np.inf*np.zeros( x.shape )
      lp[I] = alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x[I]) - beta*x[I]
      return lp
    else:
      print "*****************"
      print "gamma_logprob returning -INF"
      print "*****************"
      return -np.inf
  #else:
  #  assert False, "gamma_logprob got 0 x"

def gamma_logprob_gradient_x(x, alpha, beta):   
  #print " *************************************************************************** "
  #print " WARNING: invgamma_logprob_gradient called but it really is invgamma_logprob " 
  #print " *************************************************************************** "
  #return stats.invgamma.logpdf( x, alpha, scale = beta)
  return (alpha-1)/x - beta
  
def gamma_logprob_gradient_free_x( free_x, alpha, beta):   
  #print " *************************************************************************** "
  #print " WARNING: invgamma_logprob_gradient called but it really is invgamma_logprob " 
  #print " *************************************************************************** "
  #return stats.invgamma.logpdf( x, alpha, scale = beta)
  x = np.exp( free_x )
  
  # this is without transform
  #return (alpha-1)/x - beta
  
  # this is with transform
  return alpha-1 - x*beta
    
def beta_logprob( x, alpha, beta ):
  x = max(1e-12,x)
  x = min(x,1.0-1e-12)
  return (alpha-1)*np.log(x) + (beta-1)*np.log(1-x) + special.gammaln( alpha + beta) - special.gammaln( alpha ) - special.gammaln( beta )
  
def gp_to_z_dist_old( mu, cov ):
  #
  # want p(z), where z = mu[0]-mu[1]
  #
  # mu_z    = mu[0] - mu[1]
  # sigma_z = c[0][0] + c[1][1] + 2*c[0][1] 
  #
  
  mu_z = mu[0]-mu[1]
  sigma_z = np.sqrt( cov[0][0] + cov[1][1] - 2*cov[0][1] )
  return mu_z, sigma_z

def gp_to_z_dist( mu, cov ):
  #
  # want p(z), where z = mu[0]-mu[1]
  #
  # mu_z    = mu[0] - mu[1]
  # sigma_z = c[0][0] + c[1][1] + 2*c[0][1] 
  #
  N = len(mu)
  
  # assume last entry is xt; all the rest are proposals
  mu_z = mu[:-1]-mu[-1]
  #print cov
  
  
  var_z = np.diagonal(cov)[:-1] + cov[-1][-1] - 2*cov[-1][:-1]
  
  small = pp.find( var_z < 1e-6 )
  var_z[small] = 0
  
  sigma_z = np.sqrt(var_z)
  # print "inside gp_to_z_dist", cov, sigma_z
#   print "\t", np.diagonal(cov)[:-1], cov[-1][-1], cov[-1][:-1]
  
  if np.isnan(sigma_z).any():
  # print "=================================="
  #     print "WARNING"
  #     print "mu = ", mu
  #     print "cov = ", cov
  #     print "mu_z = ", mu_z
  #     print "var_z = ", np.diagonal(cov)[:-1] + cov[-1][-1] + 2*cov[-1][:-1]
  #     print "sigma_z = ", sigma_z
  #     print "=================================="
  #     print "assuming "
    bad = np.isnan(sigma_z)
    sigma_z[bad] = 0
  return mu_z, sigma_z
  
def normcdf( x, mu, stdev):
  if stdev > 0:
    return 0.5*(1.0 + sp.special.erf( (x-mu)/(sqrt2*stdev)))
  elif x < mu:
    return 0
  elif x > mu:
    return 1
  else:
    return 0.5

def log_logit_normal( x, mu, sigma ):
  logp = np.log(x) + np.log( 1-x )
  
  logp = - np.log(x) - np.log( 1-x ) -0.5*np.log(2.0*np.pi*sigma*sigma) - 0.5*( (np.log(x) - np.log( 1-x ) - mu)/sigma )**2    
  
  return logp
  
def logit_normal( x, mu, sigma ):
  return np.exp( log_logit_normal(x,mu,sigma))
  
def stdnormcdf( x ):
  u = 0.5*(1.0 + sp.special.erf( x/sqrt2 ))
  return u

def log_pdf_diag_mvn( x, mu, sigmas ):
  d = len(x)
  lp = - 0.5*d*np.log(2.0*np.pi) -0.5*np.log(sigmas).sum() - 0.5*(((x-mu)**2)/(sigmas**2)).sum()
  return lp

def log_pdf_full_mvn( x, mu, cov = None, invcov = None, logdet = None ):
  if cov is None:
      assert invcov is not None, "need cov or invcov"
  if invcov is None:
      invcov = sp.linalg.pinv2( cov )
      #invcov = np.linalg.pinv( cov )
    
  difx   = x-mu
  if len(x.shape) > 1 or len(mu.shape)>1:
      if len(x.shape) > 1:
        nVals = x.shape[0]
        dim   = x.shape[1] 
      else:    
        nVals = mu.shape[0]
        dim   = np.float( len(x) ) 
      malhab = (np.dot( difx, invcov ) * difx ).sum(1)
  else:
      nVals = 1
      dim = np.float( len(x) )
      malhab = np.dot( np.dot( difx, invcov ), difx )

    

  if logdet is None:
      try:
          neglogdet = np.log( np.linalg.det(cov ) ) # 
          logdet = -neglogdet
          #logdet = sum(numpy.log(numpy.linalg.svd(invcov)[1]))
      except:
          logdet = sum( np.log( np.diag( invcov ) ) )
  #print str(-0.5*dim*numpy.log( 2.0 * numpy.pi ) )
  #print str(0.5*logdet)
  #print str(malhab)
  logpdf = -0.5*dim*np.log( 2.0 * np.pi ) + 0.5*logdet - 0.5*malhab

  
  if pp.any( np.isnan( logpdf ) ) or pp.any( np.isinf( logpdf ) ):
      pdb.set_trace()
      print "********************************"
      print "********************************"
      print "log_pdf_full_mvn has inf"
      print logpdf
      print "********************************"
      print "********************************"
      return -np.inf
  return logpdf
  
def gaussian_logpdf_prec( x, mu, precisions ):
  d=x-mu
  lp = - 0.5*np.log(2.0*np.pi) + 0.5*np.log(precisions) - 0.5*precisions*d**2

  return lp
  
def gaussian_logpdf( x, mu, sigma ):
  lp = - 0.5*np.log(2.0*np.pi) -np.log(sigma)- 0.5*((x-mu)**2)/(sigma**2)

  return lp

def gaussian_logpdf_gradient_x( x, mu, sigma ):
  g = - (x-mu)/(sigma**2)

  return g
  
def gaussian_pdf( x, mu, sigma ):
  return np.exp(gaussian_logpdf( x, mu, sigma ))
                          
def lognormal_rand( mu, sigma, N = 1 ):
  log_x = mu + np.random.randn(N)*sigma
  
  return np.exp( log_x )
  
def lognormal_pdf( x, mu, sigma ):
  p = np.exp(lognormal_logpdf( x, mu, sigma ))
  return p
  
def lognormal_logpdf( x, mu, sigma ):
  small = np.log( 0.5 + 0.5*special.erf( (np.log(1e-6)-mu)/(np.sqrt(2.0)*sigma)) )
  if x.__class__ == np.ndarray:
    if sigma > 0:
      I = pp.find( x > 1e-6 )
      log_x = np.log(x[I])
      lp = small*np.ones( x.shape )
      lp[I] = -log_x - 0.5*np.log(2.0*np.pi) - np.log(sigma) - 0.5*((log_x-mu)**2)/(sigma**2)
    else:
      I = p.find( x==mu)
      lp = -np.inf*np.ones( x.shape )
      lp[I] = 0
  else:
    if sigma > 0:
      if x > 1e-6:
        log_x = np.log(x)
        lp    = -log_x - 0.5*np.log(2.0*np.pi) - np.log(sigma) - 0.5*((log_x-mu)**2)/(sigma**2)
      else:
        lp = small
    else:
      if x==mu:
        lp = 0
      else:
        lp = -np.inf
  return lp
  
def lognormal_cdf( x, mu, sigma ):
  if sigma > 0:
    small = 0.5 + 0.5*special.erf( (np.log(1e-6)-mu)/(np.sqrt(2.0)*sigma))
    if x.__class__ == np.ndarray:
      lp = np.zeros( len(x) )
      I = pp.find( x > 1e-6 )
      J = pp.find( x <= 1e-6)
      lp[I] = 0.5 + 0.5*special.erf( (np.log(x)-mu)/(np.sqrt(2.0)*sigma))
      lp[J] = small
      return lp
    else:
      if x > 1e-6:
        return 0.5 + 0.5*special.erf( (np.log(x)-mu)/(np.sqrt(2.0)*sigma))
      else:
        return small
  else:
    if x.__class__ == np.ndarray:
      logx = np.log(x+1e-6)
      lp = 0.5*np.ones( len(x))
      I1 = pp.find( logx < mu )
      I2 = pp.find( logx > mu )
      
      lp[I1] = 0
      lp[I2] = 1
      return lp
    else:
      if np.log(x) < mu:
        return 0
      elif np.log(x) > mu:
        return 1
      else:
        return 0.5
  
def lognormal_mean( mu, sigma ):
  return mu + 0.5*sigma**2
  
def lognormal_var( mu, sigma ):
  return (np.exp(sigma**2) - 1 )*np.exp( 2*mu + sigma**2 )
  
def lognormal_lower_trunc( mu, sigma, b ):
  b0 = (np.log(b) - mu)/sigma
  # print "b0 = " + str(b0)
#   print "sigma = " + str(sigma)
#   print "mu = " + str(mu)
#   print "stdnormcdf(b0) = " + str(stdnormcdf(b0))
#   print "stdnormcdf(-sigma+b0) = " + str(stdnormcdf(-sigma+b0))
#   print "np.exp( mu + sigma*sigma/2.0) = " + str(np.exp( mu + sigma*sigma/2.0))
#   if np.any( b < 0 ):
#     pdb.set_trace()
    
  z = stdnormcdf(b0)
  I = pp.find( z > 0 )
  J = pp.find( z == 0)
  
  trunc = np.zeros( z.shape )
  trunc[I] = np.exp( mu[I] + sigma[I]*sigma[I]/2.0)*stdnormcdf(-sigma[I]+b0[I])/stdnormcdf(b0[I])
  trunc[J] = np.exp( mu[J] + sigma[J]*sigma[J]/2.0)
  return trunc
  
def poisson_rand( mu, N = 1 ):
  return np.random.poisson(mu,N)
  
def poisson_logpdf( x, mu ):
  return (x-1)*np.log(mu)- gammaln(x-1) - mu
  
def bin_errors_1d( bins, true_centered_probability, samples ):
  N = len(samples)
  cnts, bins = np.histogram( samples ,bins=bins)
  
  n = cnts.sum()
  
  probability = cnts / (float(N))
  
  missed = N - n
  
  double_error = np.sum(np.abs( true_centered_probability - probability )) + missed/float(N)
  
  #pdb.set_trace()
  return double_error/2.0
    
        
def gamma_logprob( x, alpha, beta ):
  if all(x>0):
    if beta > 0:
      return alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x) - beta*x
    else:
      assert False, "Beta is zero"
  else:
    if x.__class__ == np.array:
      I = pp.find( x > 0.0 )
      lp = -np.inf*np.zeros( x.shape )
      lp[I] = alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x[I]) - beta*x[I]
      return lp
    else:
      print "*****************"
      print "gamma_logprob returning -INF"
      print "*****************"
      return -np.inf

def gen_gamma_logpdf( alpha, beta ):
  def gamma_logpdf( theta ):
    return gamma_logprob( theta, alpha, beta  )
  return gamma_logpdf
        
def gen_gamma_cdf( alpha, beta ):
  g = stats.gamma(alpha,0,1.0/beta)
  def gamma_cdf( theta ):
    return g.cdf(theta)
  return gamma_cdf
  
def lognormal_rand( mu, sigma, N = 1 ):
  log_x = mu + np.random.randn(N)*sigma
  
  return np.exp( log_x )
  
def lognormal_pdf( x, mu, sigma ):
  p = np.exp(lognormal_logpdf( x, mu, sigma ))
  return p
  
def lognormal_logpdf( x, mu, sigma ):
  small = np.log( 0.5 + 0.5*special.erf( (np.log(1e-6)-mu)/(np.sqrt(2.0)*sigma)) )
  if x.__class__ == np.ndarray:
    if sigma > 0:
      I = pp.find( x > 1e-6 )
      log_x = np.log(x[I])
      lp = small*np.ones( x.shape )
      lp[I] = -log_x - 0.5*np.log(2.0*np.pi) - np.log(sigma) - 0.5*((log_x-mu)**2)/(sigma**2)
    else:
      I = p.find( x==mu)
      lp = -np.inf*np.ones( x.shape )
      lp[I] = 0
  else:
    if sigma > 0:
      if x > 1e-6:
        log_x = np.log(x)
        lp    = -log_x - 0.5*np.log(2.0*np.pi) - np.log(sigma) - 0.5*((log_x-mu)**2)/(sigma**2)
      else:
        lp = small
    else:
      if x==mu:
        lp = 0
      else:
        lp = -np.inf
  return lp
  
def peakdet(v, delta, x = None):
    import sys
    from numpy import NaN, Inf, arange, isscalar, array
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    #v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)