
import numpy as np
import cvxpy as cp
import torch as tr
def monotonic_fourier(x, L,J,off_set,delta,alpha=1.,seed=1):
	# This function takes as input N vectors x of dimension d and outputs a applies a random point-wise non linearity on each dimension (x)_i  
	# such that f(x_i) is an increasing function of x. 
	#out(x) = b + sum a_i a_j \psi_{i,j}(x)
	# \psi_{i,j}(x) = sin( \lambda_ij^- (x+L)  )/(2L \lambda_ij^{-}) - sin( \lambda_ij^+ (x+L)  )/(2L \lambda_ij^{+})    
	# This implements the procedure in : 'A non-parametric probabilistic model for monotonic functions'

	# a : J x d  vecotr
	# b : d vector 
	# x : N x d  vector
	# out: N x d vector

	# Make a copy of the input array
	x  = np.array(x)
	N,d = x.shape
	lmbda = np.linspace(0,1,J)*delta+off_set
	# compute lmbda
	#lmbda = 1.*np.array(range(J))+1
	#lmbda *=np.pi/(2*L)
	lmbda = np.power(lmbda, alpha)
	#lmbda = np.array([1., 2., 5.])
	a = np.ones([J,d])
	state = np.random.get_state()
	#np.random.seed(seed)
	np.random.seed(1)
	a = np.random.randn(J,d)
	np.random.set_state(state)
	#lmbda = abs(lmbda)
	x += L 
	lmbda_x = np.einsum('i,nd->ind', lmbda , x )
	a_sin_x = np.einsum('id,ind->ind',a,np.sin(lmbda_x))
	a_cos_x = np.einsum('id,ind->ind',a,np.cos(lmbda_x))

	# Denominator for Psi plus.
	lmbda_ij_plus = np.expand_dims(lmbda,axis=0)+ np.expand_dims(lmbda,axis=1) 
	lmbda_ij_plus = 1./(lmbda_ij_plus)

	# Denominator for Psi_minus.
	lmbda_ij_minus = np.expand_dims(lmbda,axis=1) - np.expand_dims(lmbda,axis=0)
	# Special case for i = j. 
	lmbda_ij_minus = (1./(lmbda_ij_minus + np.eye(J)) - np.eye(J))

	# Sin((\lambda_i - \lambda_j) x) 
	#    = 1/2 (Sin(\lambda_i x)cos(\lambda_j x) + cos(\lambda_i x)sin(\lambda_j x))
	Psi_plus = 2.*np.einsum('ind,jnd,ij->nd', a_cos_x, a_sin_x, lmbda_ij_plus)
	Psi_minus = 2.*np.einsum('ind,jnd,ij->nd', a_sin_x,a_cos_x,lmbda_ij_minus)

	# Positive part of psi_{ii}
	diag_term = np.einsum('nd,d->nd',x,np.sum(a**2,axis = 0))
	out = Psi_minus - Psi_plus + diag_term 
	out /= 2*L

	# rescaling out
	out/=J*np.linalg.norm(a,axis=0)**2
	#out*=np.sqrt(delta)
	return out


def id_transform(x):

	return tfp.bijectors.Identity(x)


def covariance(kernel_fn, X, Y):
  num_rows = len(X)
  num_cols = len(Y)
  K = np.zeros((num_rows, num_cols))
  for i in range(num_rows):
    for j in range(num_cols):
      K[i, j] = kernel_fn(X[i], Y[j])

  return K


def gaussians():
  dist1 = tfd.Normal(0., 5.)
  dist2 = tfd.Normal(0., FLAGS.small_variance)
  return dist1, dist2


def plot_dists(dist1, dist2):
  a = np.arange(-10, 10, 0.01)

  plt.plot(a, dist1.prob(a), label='P')
  plt.plot(a, dist2.prob(a), label='Q')
  plt.legend()
  plt.savefig('dists{}.png'.format(FLAGS.small_variance))


def gaussian_kernel(x1, x2, gamma=0.1, height=2.):
  return height* np.exp(- gamma * np.linalg.norm(x1 - x2) ** 2)


def neg_kl_estimate_from_kernel_matrices(
    alpha, K_x_x, K_x_y, n, norm_coeff):
  # Numerical issues - specify PSD to avoid problems due to small neg values.
  K_x_x_param = cp.Parameter(
      shape=K_x_x.shape, value=K_x_x, PSD=True)
  neg_kl = cp.quad_form(alpha, K_x_x_param) / (2 * norm_coeff)
  neg_kl -= cp.sum(cp.entr(alpha))
  neg_kl += cp.sum(alpha) * (cp.log(n) - 1)
  neg_kl -= cp.sum(alpha @ K_x_y) / (norm_coeff * n)
  return neg_kl


def rkhs_model(lmbda,norm_coeff,kernel,samples1,samples2):
	n = samples1.shape[0]
	K_x_x = kernel.kernel(samples1,samples1)
	#K_x_x += tr.eye(n,dtype=K_x_x.dtype, device=K_x_x.device) * lmbda
	K_x_x = 0.5*(K_x_x+K_x_x.T)
	K_x_y = kernel.kernel(samples1,samples2)

	K_x_x = K_x_x.cpu().double().numpy()
	K_x_y = K_x_y.cpu().double().numpy()


	alpha = cp.Variable(n)
	constraints = [alpha >= 0]
	try:
		objective = cp.Minimize(neg_kl_estimate_from_kernel_matrices(
			alpha, K_x_x, K_x_y, n, norm_coeff))
		problem = cp.Problem(objective, constraints)
		problem.solve()
	except:
		K_x_x += np.eye(n) * lmbda
		objective = cp.Minimize(neg_kl_estimate_from_kernel_matrices(
			alpha, K_x_x, K_x_y, n, norm_coeff))
		problem = cp.Problem(objective, constraints)
		try:
			problem.solve()
		except:
			problem.solve(solver='SCS')
	#logging.info('Optimal value {}'.format(problem.value))
	#logging.info('True KL {}'.format(true_kl))

	alpha = np.array(alpha.value)
	
	alpha[alpha<=0] = 0.
	mask = alpha>0.
	xlogx = alpha[mask]*np.log(alpha[mask])
	est = 1 + np.sum(xlogx) + np.sum(alpha) * np.log(n/np.exp(1))
	#logging.info('Estimate {}'.format(est))

	return est





