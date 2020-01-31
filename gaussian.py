import torch as tr

FDTYPE = tr.float32
DEVICE = 'cuda'

def pow_10(x, dtype=FDTYPE,device = DEVICE): 

	return tr.pow(tr.tensor(10., dtype=dtype, device = device),x)

class BaseKernel(object):
	def __init__(self, D):
		self.D = D 
		self.params =1.
		self.isNull = False
	
	def set_params(self,params):
		raise NotImplementedError()
	def get_params(self):
		raise NotImplementedError()

	def kernel(self, X,Y):

		# Gramm matrix between vectors X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix
		raise NotImplementedError()

	def derivatives(self,X,Y):
		# Computes the Hadamard product between the gramm matrix of (Y, basis) and the matrix K
		# Input:
		# X 	: N by d matrix of data points
		# Y 	: M by d matrix of basis points		
		# output: N by M matrix   K had_prod Gramm(Y, basis)
		raise NotImplementedError()

	def __add__(self, other):
		if other.D != self.D:
			raise NameError('Dimensions of kernels do not match !')
		else:
			new_kernel = CombinedKernel(self.D, [self, other])
			return new_kernel


class CombinedKernel(BaseKernel):
	def __init__(self,D,  kernels):
		BaseKernel.__init__(self, D)
		self.kernels = kernels

	def kernel(self, X,Y):
		K = 0.
		for kernel in self.kernels:
			K += kernel.kernel(X, Y)
		return K 


	def derivatives(self,X,Y):

		raise NotImplementedError()



class Gaussian(BaseKernel):
	def __init__(self, D,  log_sigma, dtype = FDTYPE, device = DEVICE):
		BaseKernel.__init__(self, D)
		self.params  = log_sigma
		self.dtype = dtype
		self.device = device
		self.adaptive=  False
		self.params_0 = log_sigma  

	# def exp_params(self,sigma):
	# 	return pow_10(sigma)
	def get_exp_params(self):
		return pow_10(self.params, dtype= self.dtype, device = self.device)
	def update_params(self,log_sigma):
		self.params = log_sigma
	# def set_params(self,params):
	# 	# stores the inverse of the bandwidth of the kernel
	# 	if isinstance(sigma, float):
 #            self.params  = tr.from_numpy(sigma)
 #        elif type(sigma)==tr.Tensor:
 #            self.params = sigma
 #        else:
 #            raise NameError("sigma should be a float or tf.Tensor")
	# 	self.params = params

	# def get_params(self):
	# 	# returns the bandwidth of the gaussian kernel
	# 	return self.params


	def square_dist(self, X, Y):
		# Squared distance matrix of pariwise elements in X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._square_dist( X, Y)

	def kernel(self, X,Y):

		# Gramm matrix between vectors X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._kernel(self.params,X, Y)

	def derivatives(self, X,Y):
		return self._derivatives(self.params,X,Y)

	def dkdy(self, X,Y):
		return self._dkdy(self.params,X,Y)

	def dkdy_dot_dkdy(self,X,Y):
		return self._dkdy_dot_dkdy(self.params,X,Y)

	def dkdxdy_dot_dkdxdy(self,X,Y):
		return self._dkdxdy_dot_dkdxdy(self.params,X,Y)
	def dkdxdy(self,X,Y,mask=None):
		return self._dkdxdy(self.params,X,Y,mask=mask)
	def dkdxdy_sym(self,X,mask):
		return self._dkdxdy_sym(self.params,X,mask)
# Private functions 

	def _square_dist(self,X, Y):
		n_x,d = X.shape
		n_y,d = Y.shape
#		dist = -2*tr.einsum('mr,nr->mn',X,Y) + tr.einsum('m,n->mn',tr.sum(X**2,1), tr.ones([ n_y],dtype=self.dtype, device = self.device)) +  tr.einsum('m,n->mn', tr.ones([ n_x],dtype=self.dtype, device = self.device),tr.sum(Y**2,1)) 
		dist = -2*tr.einsum('mr,nr->mn',X,Y) + tr.sum(X**2,1).unsqueeze(-1).repeat(1,n_y) +  tr.sum(Y**2,1).unsqueeze(0).repeat(n_x,1) #  tr.einsum('m,n->mn', tr.ones([ n_x],dtype=self.dtype, device = self.device),tr.sum(Y**2,1)) 

		return dist 

	def _kernel(self,log_sigma,X,Y):
		N,d = X.shape
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		tmp = self._square_dist( X, Y)
		dist = tr.max(tmp,tr.zeros_like(tmp))
		if self.adaptive:
			ss = tr.mean(dist).clone().detach()
			dist = dist/(ss+1e-5)
		return  tr.exp(-0.5*dist/sigma)

	def _dkdy(self,log_sigma,X,Y):

		# X : [M,T]
		# Y : [N,R]

		# dkdxdy ,   dkdxdy2  = [M,N,T,R]  
		# dkdy2 = [M,N,R]
		# dkdY = [M,N,R]
		# dkdx = [M,N,T]
		# gram =  [M,N]
		N,d = X.shape
		#assert d==1 
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		gram = self._kernel(log_sigma,X, Y)

		D = (X.unsqueeze(1) - Y.unsqueeze(0))/sigma
		 
		#I  = tr.ones( D.shape[-1],dtype=self.dtype, device = self.device)/sigma

		dkdy = tr.einsum('mn,mnr->mnr', gram,D)
		
		

		return dkdy, gram

	def _dkdy_dot_dkdy(self,log_sigma,X,Y):
		M,d = X.shape
		N,_ = Y.shape
		#assert d==1 
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		gram = self._kernel(log_sigma,X, Y)
		tmp = tr.einsum('md,kd->mk',X,X)
		out = tr.einsum('mn,kn->mk',gram,gram)
		out = tr.einsum('mk,mk->mk',out,tmp)
		
		tmp = tr.einsum('nd,nd->n',Y,Y)
		tmp = tr.einsum('mn,n->mn',gram,tmp)
		out += tr.einsum('mn,kn->mk',gram,tmp)
		
		tmp = tr.einsum('md,nd->mn',X,Y)
		tmp = tr.einsum('mn,mn->mn',gram,tmp)
		tmp = tr.einsum('mn,kn->mk',gram,tmp)
		out -= (tmp + tmp.t())
		out = out/sigma**2
		return out, gram


	def _dkdxdy_dot_dkdxdy(self,log_sigma,X,Y):

		N,d = X.shape
		#assert d==1 
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		gram = self._kernel(log_sigma,X, Y)

		KK = tr.einsum('mn,kn->mk',gram,gram)
		Identity = tr.eye(d, dtype=self.dtype, device=self.device)
		term_1 = tr.einsum('mk,ij->mkij',KK,Identity)/sigma**2
		XX = tr.einsum('ki,kj->kij',X,X)
		term_2 = tr.einsum('mk,kij->mkij',KK,XX)
		
		tmp_1 = tr.einsum('mn,ni->mni', gram,Y)
		tmp = tr.einsum('kn,mni->mki',gram,tmp_1)
		term_3 = tr.einsum('mki,kj->mkij',tmp,X)

		tmp_2 = tr.einsum('kn,nj->knj', gram,Y)
		tmp = tr.einsum('mn,knj->mkj',gram,tmp_2)
		term_4 = tr.einsum('mkj,ki->mkij',tmp,X)

		term_5 = tr.einsum('mni,knj->mkij',tmp_1,tmp_2)

		term_6 =  (term_2 - term_3 - term_4 + term_5)/sigma**3
		term_6 += term_6.permute(1,0,3,2)

		XX = tr.einsum('mj,kj->mk',X,X)
		XY = tr.einsum('mj,nj->mn',X,Y)
		YY = tr.einsum('nj,nj->n',Y,Y)

		XY = XY-0.5*YY.unsqueeze(0)

		XX_i = tr.einsum('mi,kj->mkij',X,X)
		
		tmp_a = tr.einsum('mn,mn->mn',gram,XY)
		term_7a = tr.einsum('mn,kn->mk',tmp_a,gram)
		tmp = tr.einsum('mi,kj->mkij',X,X)
		term_7a = tr.einsum('mk,mkij->mkij',term_7a,tmp)

		tmp_0 = tr.einsum('kn,nj->knj',gram,Y)
		tmp_2 = tr.einsum('mn,knj->mkj',tmp_a,tmp_0)
		term_7b = tr.einsum('mkj,mi->mkij',tmp_2,X)
		term_7c = tr.einsum('mki,kj->mkij',tmp_2,X)
		tmp = tr.einsum('mn,ni->mni',tmp_a,Y)	
		term_7d = tr.einsum('mni,knj->mkij',tmp,tmp_0)

		term_7 = term_7a - (term_7b+term_7c) + term_7d
		term_7 += term_7.permute(1,0,3,2)



		term_8_a = tr.einsum('mn,knj->mkj',gram,tmp_0)
		term_8_a = tr.einsum('mkj,mk->mkj',term_8_a,XX)
		term_8_a = tr.einsum('mkj,mi->mkij',term_8_a,X)
		term_8_a += term_8_a.permute(1,0,3,2)
		term_8_b = tr.einsum('mni,knj->mkij',tmp_0,tmp_0)
		term_8_b = tr.einsum('mkij,mk->mkij',term_8_b,XX)
		term_8_c = tr.einsum('mk,mk->mk',KK,XX)
		term_8_c =  tr.einsum('mk,mi->mki',term_8_c,X)
		term_8_c = tr.einsum('mki,kj->mkij',term_8_c,X)
		term_8 = term_8_c + term_8_b - term_8_a
		term_9 = term_8 - term_7
		term_9 = term_9/sigma**4




		out =  term_1 -term_6+ term_9
		return out, gram

	def _dkdxdy(self,log_sigma,X,Y,mask=None):
		# X : [M,T]
		# Y : [N,R]

		# dkdxdy ,   dkdxdy2  = [M,N,T,R]  
		# dkdy2 = [M,N,R]
		# dkdY = [M,N,R]
		# dkdx = [M,N,T]
		# gram =  [M,N]
		N,d = X.shape
		#assert d==1 
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		gram = self._kernel(log_sigma,X, Y)

		D = (X.unsqueeze(1) - Y.unsqueeze(0))/sigma
		 
		I  = tr.ones( D.shape[-1],dtype=self.dtype, device = self.device)/sigma

		dkdy = tr.einsum('mn,mnr->mnr', gram,D)
		dkdx = -dkdy



		if mask is None:
			D2 = tr.einsum('mnt,mnr->mntr', D, D)
			I  = tr.eye( D.shape[-1],dtype=self.dtype, device = self.device)/sigma
			dkdxdy = I - D2
			dkdxdy = tr.einsum('mn, mntr->mntr', gram, dkdxdy)
		else:
			#I  = tr.eye( d,dtype=self.dtype, device = self.device)/sigma
			D_masked = tr.einsum('mnt,mt->mn', D, mask)
			D2 = tr.einsum('mn,mnr->mnr', D_masked, D)
			#I_masked = tr.einsum('mt,tr->mr', mask,I)
			dkdxdy =  tr.einsum('mn,mr->mnr', gram, mask)/sigma  -tr.einsum('mn, mnr->mnr', gram, D2)
			dkdx = tr.einsum('mnt,mt->mn',dkdx,mask)

		#dkdxdy = tr.einsum('mntr->mn',dkdxdy)
		#dkdxdy2 = tr.einsum('mntr->mn',dkdxdy2)
		#dkdy2 = tr.einsum('mnr->mn', dkdy2)
		#dkdy = tr.einsum('mnr->mn', dkdy)
		#dkdx = tr.einsum('mnr->mn', dkdx)

		return dkdxdy, dkdx, gram


	def _dkdxdy_sym(self,log_sigma,X,mask):
		# X : [M,T]
		# Y : [N,R]

		# dkdxdy ,   dkdxdy2  = [M,N,T,R]  
		# dkdy2 = [M,N,R]
		# dkdY = [M,N,R]
		# dkdx = [M,N,T]
		# gram =  [M,N]
		N,d = X.shape
		#assert d==1 
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		
		#X_masked = tr.einsum('mt,mt->m',X,mask)
		gram = self._kernel(log_sigma,X, X)
		D = (X.unsqueeze(1) - X.unsqueeze(0))/sigma
		 
		#I  = tr.ones( D.shape[-1],dtype=self.dtype, device = self.device)/sigma

		#dkdy = tr.einsum('mn,mnr->mnr', gram,D)
		#dkdx = -dkdy

			#I  = tr.eye( d,dtype=self.dtype, device = self.device)/sigma
			#D_masked = tr.einsum('mnt,mt->mn', D, mask)
		D_masked = tr.einsum('mnt,mt->mn', D, mask)
		D_masked_2 = tr.einsum('mnt,nt->mn', D, mask)
		D2 = tr.einsum('mn,mn->mn', D_masked, D_masked_2)
			#I_masked = tr.einsum('mt,tr->mr', mask,I)
		dkdxdy =  tr.einsum('mr,nr->mn',mask,mask) * gram/sigma  -tr.einsum('mn, mn->mn', gram, D2)

		#dkdxdy = tr.einsum('mntr->mn',dkdxdy)
		#dkdxdy2 = tr.einsum('mntr->mn',dkdxdy2)
		#dkdy2 = tr.einsum('mnr->mn', dkdy2)
		#dkdy = tr.einsum('mnr->mn', dkdy)
		#dkdx = tr.einsum('mnr->mn', dkdx)

		return dkdxdy








	def _derivatives(self,log_sigma,X,Y):
		# X : [M,T]
		# Y : [N,R]

		# dkdxdy ,   dkdxdy2  = [M,N,T,R]  
		# dkdy2 = [M,N,R]
		# dkdY = [M,N,R]
		# dkdx = [M,N,T]
		# gram =  [M,N]
		N,d = X.shape
		#assert d==1 
		sigma = pow_10(log_sigma,dtype= self.dtype, device = self.device)
		gram = self._kernel(log_sigma,X, Y)

		D = (X.unsqueeze(1) - Y.unsqueeze(0))/sigma
		 
		I  = tr.ones( D.shape[-1],dtype=self.dtype, device = self.device)/sigma

		dkdy = tr.einsum('mn,mnr->mnr', gram,D)
		dkdx = -dkdy



		D2 = D**2
		dkdy2 = D2-I

		dkdy2 = tr.einsum('mn,mnr->mnr', gram,dkdy2)

		
		D2 = tr.einsum('mnt,mnr->mntr', D, D)

		I  = tr.eye( D.shape[-1],dtype=self.dtype, device = self.device)/sigma
		dkdxdy = I - D2
		dkdxdy = tr.einsum('mn, mntr->mntr', gram, dkdxdy)

		h =  1./sigma - D**2


		hD = tr.einsum('mnt,mnr->mntr',D,h)

		hD2 = 2.*tr.einsum('mnr,tr->mntr',D,I)

		dkdxdy2 = hD2 + hD

		dkdxdy2 = tr.einsum('mn,mntr->mntr', gram, dkdxdy2)


		#dkdxdy = tr.einsum('mntr->mn',dkdxdy)
		#dkdxdy2 = tr.einsum('mntr->mn',dkdxdy2)
		#dkdy2 = tr.einsum('mnr->mn', dkdy2)
		#dkdy = tr.einsum('mnr->mn', dkdy)
		#dkdx = tr.einsum('mnr->mn', dkdx)

		return dkdxdy, dkdxdy2, dkdx, dkdy, dkdy2, gram





























