import numpy as np
#from scipy.special import logsumexp
#from scipy.integrate import quad
#from autograd.scipy.stats import norm
#from multiprocessing import Pool
#from nystrom_kexpfam.density import rings_log_pdf_grad, rings_sample, rings_log_pdf
#from nystrom_kexpfam.data_generators.Gaussian import GaussianGrid
#from Utils import support_1d
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
import h5py as h5
from scipy.stats import truncnorm as tnorm
from scipy.linalg import expm
#from autograd import elementwise_grad

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans, SpectralClustering

def apply_whiten(data):
    
    mean = data.mean(0)
    data = data - data.mean(0)
    u, s, vt = np.linalg.svd(data[:10**4])
    W = vt.T/s * np.sqrt(u.shape[0])
    data = np.dot(data, W)
    return data, W, mean, s

def inv_whiten(data, W, mean):
    return data.dot(np.linalg.inv(W)) + mean

def apply_scale(data):
    
    mean = data.mean(0)
    data = data - data.mean(0)
    std  = data.std(0)
    data /= std

    return data, std, mean

def inv_scale(data, std, mean):
    return data*std + mean

def apply_itanh(data):
    
    m  =  data.min(axis=0)
    data -= m
    ptp = data.ptp(axis=0) 
    data /= ptp / 0.98 / 2
    data -= 0.98

    data = np.arctanh(data)

    m2 = data.mean(0)
    data -= m2

    return data, ptp, m, m2

def inv_itanh(data, ptp, m, m2):

    data += m2
    data = np.tanh(data)
    data += 0.98
    data *= ptp / 0.98/2
    data += m

    return data

    


class Dataset(object):

    def sample(self, n):
        raise NotImplementedError

    def sample_two(self, n1, n2):
        raise NotImplementedError


class ToyDataset(Dataset):

    def sample(self, n):
        raise NotImplementedError

    def sample_two(self, n1, n2):
        return self.sample(n1), self.sample(n2)

    def logpdf_multiple(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        return support_1d(self.logpdf_multiple, x)

    def log_pdf(self, x):
        return support_1d(self.logpdf_multiple, x)

    def log_pdf_multile(self, x):
        return self.logpdf_multiple(x)


    def dlogpdf(self, x):
        return grad_multiple(x)

    def grad_multiple(self, x):
        raise NotImplementedError

    def grad(self, x):
        return self.grad_multiple(self.logpdf, x)

    def score(self, x):
        return -0.5*self.grad_multiple(x**2,1)

# class Door(ToyDataset):
#     def __init__(self, width=2.0, D=2, h_min = 0.5, h_rate = 2.0):
        
#         self.width = width
#         self.D = D
#         self.h_min = h_min
#         self.h_rate = h_rate
#         self.has_grad = True
#         self.name = "door"

#     def sample(self, N):
#         s = np.zeros((N, 2))
#         s[:,0] = np.random.randn(N)*self.width
#         s[:,1] = (self.h_rate * np.cos(0.7*s[:,0])**2 + self.h_min) * np.random.randn(N) 
#         return s

#     def logpdf_multiple(self, x):
        
#         logpdf  = norm.logpdf(x[:,0], 0, self.width)
#         logpdf += norm.logpdf(x[:,1], 0, (self.h_rate * np.cos(0.7*x[:,0])**2  + self.h_min))
#         return logpdf

#     def grad_multiple(self, x):
        
#         #g = elementwise_grad(self.logpdf_multiple)
#         #return g(x)
#         return 0

# class Spiral(ToyDataset):
    
#     def __init__(self, sigma=0.5, D = 2, eps=1, r_scale=1.5, starts=np.array([0.0,2.0/3,4.0/3]) * np.pi, 
#                 length=np.pi):

#         self.sigma = sigma
#         self.L= length
#         self.r_scale = r_scale
#         self.D = D
#         self.eps = eps # add a small noise at the center of spiral
#         self.starts= starts
#         self.nstart= len(starts)
#         self.name = "spiral"
#         self.has_grad = False

#     def _branch_params(self, a, start):
        
#         n = len(a)
#         a = self.L * ((a)**(1.0/self.eps))+ start
#         r = (a-start)*self.r_scale
        
#         m = np.zeros((n, self.D))
#         s = np.ones((n, self.D)) * self.sigma
        
#         m[:,0] = r * np.cos(a)
#         m[:,1] = r * np.sin(a)
#         s[:,:2] = (a[:,None]-start)/self.L * self.sigma + 0.1

#         return m, s

#     def _branch_params_one(self, a, start):
        
#         a = self.L * ((a)**(1.0/self.eps))+ start
#         r = (a-start)*self.r_scale
        
#         m = np.zeros((self.D))
#         s = np.ones((self.D)) * self.sigma
        
#         m[0] = r * np.cos(a)
#         m[1] = r * np.sin(a)
#         s[:2] = (a-start)/self.L * self.sigma

#         return m, s

#     def sample(self, n):
        
#         data = np.zeros((n+self.nstart, self.D))
#         batch_size = np.floor_divide(n+self.nstart,self.nstart)
        
#         for si, s in enumerate(self.starts):
#             m = np.floor_divide(n,self.nstart)
#             data[si*batch_size:(si+1)*batch_size] = self.sample_branch(batch_size, s)
#         return  data[:n,:]

        
        
#     def sample_branch(self, n, start):
        
#         a = np.random.uniform(0,1,n)

#         m, s = self._branch_params(a, start) 

#         data = m + np.random.randn(n, self.D) * s
#         return data

#     def _conditional_pdf(self, a, x):
        
#         n = x.shape[0]
#         p = np.array((n,self.nstart))

#         for si, s in enumerate(self.starts):
            
#             m, s = self._branch_params(a, s)
#             pdf[:,si] = norm.logpdf(x, loc = m, scale = s).sum(1)
#             pdf[:,si] -= np.log(self.nstart)

#         return np.sum(np.exp(pdf), 1)

#     def _conditional_pdf_one(self, a, x):
        
#         pdf = np.zeros((self.nstart))

#         for si, s in enumerate(self.starts):
            
#             m, s = self._branch_params_one(a, s)
#             pdf[si] = norm.logpdf(x, loc = m, scale = s).sum()
#             pdf[si] -= np.log(self.nstart)

#         return np.sum(np.exp(pdf))

#     def _conditional_dpdf_one_dim(self, a, x, D):

#         dpdf = np.zeros((self.nstart))
        
#         for si, s in enumerate(self.starts):
            
#             m, s = self._branch_params_one(a, s)
#             dpdf[si] = np.exp(norm.logpdf(x, loc = m, scale = s).sum()) * ( - x[D] + m[D]) / s[D]**2
#             dpdf[si] /= self.nstart

#         return dpdf.sum()

#     def pdf_one(self, x, *args, **kwargs):
        
#         return quad(self._conditional_pdf_one, 0, 1, x, *args, **kwargs)[0]

#     def dpdf_one(self, x, *args, **kwargs):
        
#         dpdf = np.zeros(self.D)
#         for d in range(self.D):
#             dpdf[d] = quad(self._conditional_dpdf_one_dim, 0, 1, (x, d), *args, **kwargs)[0]
#         return dpdf

#     def grad_one(self, x, *args, **kwargs):
        
#         return self.dpdf_one(x, *args, **kwargs) / self.pdf_one(x, *args, **kwargs)


# class Funnel(ToyDataset):
    
#     def __init__(self, sigma=2.0, D=2, lim=10.0):
    
#         self.sigma=sigma
#         self.D=D
#         self.lim=lim
#         self.low_lim = 0.000
#         self.thresh   = lambda x: np.clip(np.exp(-x), self.low_lim, self.lim)
#         self.name="funnel"
#         self.has_grad = True
        
        
#     def sample(self, n):
        
#         data = np.random.randn(n, self.D)
#         data[:,0]  *= self.sigma
#         v =  self.thresh(data[:,0:1])
#         data[:,1:] = data[:,1:] * np.sqrt(v)
#         return data
    
#     def grad_multiple(self, x):
        
#         N = x.shape[0]
#         grad = np.zeros((N, self.D))
#         x1 = x[:,0]
        
#         v = np.exp(-x1)
        
#         dv  = -1*v
#         dlv = -np.ones_like(v)
        
#         dv[(v) < self.low_lim] = 0
#         dv[(v) > self.lim] = 0
        
#         dlv[(v) < self.low_lim] = 0
#         dlv[(v) > self.lim] = 0
        
#         grad[:,0] = -x1/self.sigma**2 - (self.D-1)/2.0 * dlv - 0.5*(x[:,1:]**2).sum(1) * (-dv)/v/v
#         grad[:,1:]= - x[:,1:] / self.thresh(x1)[:,None]
#         return grad
    
#     def logpdf_multiple(self, x):
#         v = self.thresh(x[:,0])
#         return norm.logpdf(x[:,0], 0, self.sigma) + norm.logpdf(x[:,1:], 0, np.sqrt(v)[:,None]).sum(1)

# class Ring(ToyDataset):

#     def __init__(self, sigma=0.2, D=2, nring = 1):

#         assert D >= 2
        
#         self.sigma = sigma
#         self.D = D
        
#         self.radia = np.array([5])
#         self.name  = "ring"
#         self.has_grad = True
        
#     def grad_multiple(self, X):
#         return rings_log_pdf_grad(X, self.sigma, self.radia)

#     def logpdf_multiple(self, X):
#         return rings_log_pdf(X, self.sigma, self.radia)

#     def sample(self, N):
#         samples = rings_sample(N, self.D, self.sigma, self.radia)
#         return samples

# class Multiring(ToyDataset):

#     def __init__(self, sigma=0.2, D=2):

#         assert D >= 2
        
#         self.sigma = sigma
#         self.D = D
#         self.radia = np.array([1, 3, 5])
#         self.name  = "multiring"
#         self.has_grad = True
        
#     def grad_multiple(self, X):
#         return rings_log_pdf_grad(X, self.sigma, self.radia)

#     def logpdf_multiple(self, X):
#         return rings_log_pdf(X, self.sigma, self.radia)

#     def sample(self, N):
#         samples = rings_sample(N, self.D, self.sigma, self.radia)
#         return samples


# class Grid(ToyDataset):

#     def __init__(self, sigma=0.5, D=2, sep=8):

#         assert D >= 2
        
#         self.sigma = sigma
#         self.D = D
#         self.name  = "grid"
#         self.has_grad = True
#         np.random.seed(1)
#         self._g = GaussianGrid(D, sigma, sep=sep)
        
#     def grad_multiple(self, X):
#         return self._g.grad_multiple(X)

#     def logpdf_multiple(self, X):
#         return self._g.log_pdf_multiple(X)

#     def sample(self, N):
#         return self._g.sample(N)

# class Cosine(ToyDataset):

#     def __init__(self, D=2, sigma=1.0, xlim = 4, omega=2, A=3):

#         assert D >= 2
        
#         self.sigma = sigma
#         self.xlim  = xlim
#         self.A = A
#         self.D = D
#         self.name="cosine"
#         self.has_grad=True
#         self.omega = omega
        
#     def grad_multiple(self, X):

#         grad = np.zeros_like(X)
#         x0= X[:,0]
#         m = self.A*np.cos(self.omega*x0)
#         #grad[:,0] = -X[:,0]/self.xlim**2 - (X[:,1]-m)/self.sigma**2 * self.A * np.sin(self.omega*X[:,0])*self.omega
#         grad[:,0] = -(X[:,1]-m)/self.sigma**2 * self.A * np.sin(self.omega*X[:,0])*self.omega
#         grad[:,1] = -(X[:,1]-m)/self.sigma**2
#         grad[np.abs(x0)>self.xlim,:]=0
#         if self.D>2:
#             grad[:,2:] = -X[:, 2:]
        
#         return grad

#     def logpdf_multiple(self, X):
        
#         x0 = X[:,0]
#         x1_mu = self.A * np.cos(self.omega*x0)
        
#         #logpdf = norm.logpdf(x0, 0, self.xlim)
#         logpdf  = -np.ones(X.shape[0])*np.log(2*self.xlim)
#         logpdf += norm.logpdf(X[:,1], x1_mu, self.sigma)
#         logpdf += np.sum(norm.logpdf(X[:,2:], 0, 1), -1)

#         logpdf[np.abs(x0)>self.xlim] = -np.inf
        
#         return logpdf

#     def sample(self, N):
#         x0 = np.random.uniform(-self.xlim, self.xlim, N)
#         x1 = self.A * np.cos(self.omega*x0)

#         x = np.random.randn(N, self.D)
#         x[:,0] = x0
#         x[:,1] *= self.sigma
#         x[:,1] += x1
#         return x

# class Uniform(ToyDataset):

#      def __init__(self, D=2,lims = 3):
#          self.lims = lims
#          self.D = D
#          self.has_grad = True

#      def sample(self,n):
#          return 2*(np.random.rand(n, self.D) - 0.5) * self.lims

#      def logpdf_multiple(self, x):
#          pdf = - np.ones(x.shape[0]) * np.inf
#          inbounds = np.all( (x<self.lims) * ( x > -self.lims), -1)
#          pdf[inbounds] = -np.log((2*self.lims)**self.D)
#          return pdf
         
#      def grad_multiple(self, x):
         
#          return np.zeros_like(x) 

# class Banana(ToyDataset):
    
#     def __init__(self, bananicity = 0.2, sigma=2, D=2):
#         self.bananicity = bananicity
#         self.sigma = sigma
#         self.D = D
#         self.name = "banana"
#         self.has_grad = True

#     def logpdf_multiple(self,x):
#         x = np.atleast_2d(x)
#         assert x.shape[1] == self.D
#         logp =  norm.logpdf(x[:,0], 0, self.sigma) + \
#                 norm.logpdf(x[:,1], self.bananicity * (x[:,0]**2-self.sigma**2), 1)
#         if self.D > 2:
#             logp += norm.logpdf(x[:,2:], 0,1).sum(1)

#         return logp

#     def sample(self, n):
        
#         X = np.random.randn(n, self.D)
#         X[:, 0] = self.sigma * X[:, 0]
#         X[:, 1] = X[:, 1] + self.bananicity * (X[:, 0] ** 2 - self.sigma**2)
#         if self.D > 2:
#             X[:,2:] = np.random.randn(n, self.D - 2)
        
#         return X

#     def grad_multiple(self, x):
        
#         x = np.atleast_2d(x)
#         assert x.shape[1] == self.D

#         grad = np.zeros(x.shape)
        
#         quad = x[:,1] - self.bananicity * (x[:,0]**2 - self.sigma**2)
#         grad[:,0] = -x[:,0]/self.sigma**2 + quad * 2 * self.bananicity * x[:,0]
#         grad[:,1] = -quad
#         if self.D > 2:
#             grad[:,2:] = -x[:, 2:]
#         return grad

def clean_data(data, cor=0.98):

    C = np.abs(np.corrcoef(data.T))
    B = np.sum(C>cor, axis=1)
    while np.any(B>1):
        col_to_remove = np.where(B>1)[0][0]
        data = np.delete(data, col_to_remove, axis=1)
        C = np.corrcoef(data.T)
        B = np.sum(C>cor, axis=1)

    return data

class RealDataset(Dataset):

    def __init__(self, idx=None, N=None, valid_thresh=0.0, noise_std = 0.0, nkde=0,
                ntest = None, seed=0, permute=True, itanh=False, whiten=True, dequantise=False, 
                N_train = None):
        
        np.random.seed(seed) 
        np.random.shuffle(self.data)

        if idx is not None:
            self.data = self.data[:,idx]
        if N is not None:
            self.data = self.data[:N]
        else:   
            N = self.data.shape[0]
            self.data = self.data[:N]
        
        if dequantise:
            for d in range(self.data.shape[1]):
                diff = np.median(np.diff(np.unique(self.data[:,d])))
                n = self.data.shape[0] 
                self.data[:,d] += (np.random.rand(n)*2-1) * diff * 1
      
        self.nround   = 0
        self.pointer  = 0

        self.itanh    = itanh
        self.whiten   = whiten

        self.idx = idx
        self.noise_std = noise_std
        
        if permute:
            idx = np.random.permutation(self.data.shape[1])
            self.data = self.data[:,idx]
        else:
            idx = np.arange(self.data.shape[1])

        if whiten:
            self.data, self.W, self.mean, self.s = apply_whiten(self.data)
        else:
            self.W = np.eye(self.data.shape[1])
            self.mean = np.zeros(self.data.shape[1])
            self.s = np.ones(self.data.shape[1])
        
        self.data += np.random.randn(*self.data.shape) * noise_std

        if itanh:
            self.data, self.ptp, self.min, self.mean2 = apply_itanh(self.data)
        
        if ntest is None:
            self.ntest = int(N * 0.1)
            ntest = self.ntest
        else:
            self.ntest = ntest
        
        self.all_data  = self.data.copy()
        if ntest == 0:
            self.test_data = self.all_data[:0] 
            self.data = self.all_data[:]
        else:
            self.test_data = self.all_data[-ntest:]
            self.data = self.all_data[:-ntest]

        nvalid = min(int(self.data.shape[0]*0.1) , 1000)
        self.nvalid = nvalid
        self.valid_data = self.data[-nvalid:]
        self.data = self.data[:-nvalid]

        n = self.data.shape[0]
        self.data = self.data[:n]

        self.N, self.D = self.data.shape
        self.valid_thresh = valid_thresh
        self.update_data()

        if idx is None:
            self.idx = range(self.D)
        else:
            self.idx = idx
        

        if N_train is not None:

            self.N_prop = N_train*1.0 / self.N
            ndata = N_train
            self.data = self.data[:ndata]

            nvalid = max(300, int(np.floor(self.nvalid * self.N_prop)))
            self.valid_data = self.valid_data[:nvalid]
            self.update_data()
        else:
            self.N_prop = None

        
        self.nkde=nkde
        if nkde:
            self.kde_logp, self.valid_kde_logp = self.fit_kde(nkde)
    
    def fit_kde(self, ntrain):

        # use grid search cross-validation to optimize the bandwidth
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(self.data[:ntrain])

        kde = grid.best_estimator_
        kde_logp = kde.score_samples(self.data)
        valid_kde_logp = kde.score_samples(self.valid_data)
        self.kde = kde
        return kde_logp , valid_kde_logp

    def update_data(self):
        
        if self.N < 10**4:
            self.close_mat = euclidean_distances(self.data) > self.valid_thresh
        self.N = self.data.shape[0]
        self.nvalid = self.valid_data.shape[0]
        self.ntest  = self.test_data.shape[0]

    def valid_idx(self, idx):

        return np.where(np.all(self.close_mat[idx], 0))[0]

    def sample(self, n, add_noise=False):

        n = min(n, self.N)
        idx = np.random.choice(self.N,n,replace=False)
        d = self.data[idx]

        return d

    def sample_two(self, n1, n2, add_noise=False):


        if self.N < 10**4:

            idx = np.random.choice(self.N,n1,replace=False)
            s1 = self.data[idx]

            valid_idx = self.valid_idx(idx)
            idx = np.random.choice(valid_idx, n2)
            s2 = self.data[idx]

        else:
            
            n = n1+n2
            idx = np.random.choice(self.N,n,replace=True)
            s = self.data[idx]
            s1 = s[:n1]
            s2 = s[n1:]

        return s1, s2

    def stream(self, n, add_noise=False):

        d = np.take(self.data, range(self.pointer, self.pointer+n), mode="wrap", axis=0)
        if self.nkde:
            p = np.take(self.kde_logp, range(self.pointer, self.pointer+n), mode="wrap", axis=0)
        else:
            p = None
        self.increment_pointer(n)

        return d, p

    def stream_two(self, n1, n2, add_noise=False):
        
        n = n1 + n2
        d = np.take(self.data, range(self.pointer, self.pointer+n), mode="wrap", axis=0)
        
        s1 = d[:n1]
        s2 = d[n1:]
        if self.nkde:
            p = np.take(self.kde_logp, range(self.pointer, self.pointer+n), mode="wrap", axis=0)
            p1 = p[:n1]
            p2 = p[n1:]
        else:
            p1, p2 = None, None

        self.increment_pointer(n)

        return s1, s2, p1, p2

    def increment_pointer(self, n):
        
        self.pointer += n
        if self.pointer / self.N - self.nround > 0:
            self.nround += 1
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            if self.nkde:
                self.kde_logp = self.kde_logp[idx]

    def itrans(self, data):
        
        if self.itanh:
            data = inv_itanh(data, self.ptp, self.min, self.mean2)

        if self.whiten:
            data = inv_whiten(data, self.W, self.mean)
            
        data = data[:, np.argsort(self.idx)]

        return data
        
    def trans(self, data):
        # assuming only whitening 

        data = data[:, self.idx]
        if self.whiten:
            data = (data - self.mean).dot(self.W)

        return data

class WhiteWine(RealDataset):
    
    def __init__(self, *args, **kwargs):
        self.data = np.loadtxt("data/winequality-white.csv", delimiter=";", skiprows=1)[:,:-1]
        self.name="WhiteWine"
        
        super(WhiteWine, self).__init__(*args, **kwargs)
    
class RedWine(RealDataset):
    
    def __init__(self, *args, **kwargs):
        self.data = np.loadtxt("data/winequality-red.csv", delimiter=";", skiprows=1)[:,:-1]
        self.name="RedWine"

        super(RedWine, self).__init__(*args, **kwargs)

class HepMass(RealDataset):
    
    def __init__(self, *args, **kwargs):
        self.data = np.load("data/hepmass.npz")["data"]
        self.data = self.data[self.data[:,0]==1,:]
        self.data = np.delete(self.data, [0,6,10,14,18,22], axis=1)
        self.name="HepMass"

        super(HepMass, self).__init__(ntest = 174900, *args, **kwargs)

class MiniBoone(RealDataset):
    
    def __init__(self, *args, **kwargs):
        self.data = np.load("data/miniboone.npy")
        self.name="MiniBoone"

        super(MiniBoone, self).__init__(*args, **kwargs)

class Parkinsons(RealDataset):
    def __init__(self, cor=0.98, *args, **kwargs):
        self.data = np.loadtxt("data/parkinsons_updrs.data", delimiter=",", skiprows=1)[:,3:]
        self.name="Parkinsons"
        self.data = clean_data(self.data, cor=cor)
        
        super(Parkinsons, self).__init__(*args, **kwargs)
    
class Gas(RealDataset):
    def __init__(self, cor=0.98, *args, **kwargs):
        self.data = np.array(np.load("data/ethylene_CO.pickle"))[:,3:]
        self.name="Gas"
        self.data = clean_data(self.data, cor=cor)

        super(Gas, self).__init__(*args, **kwargs)

class Power(RealDataset):
    def __init__(self, cor=0.98, *args, **kwargs):
        self.data = np.load("data/power.npy")
        self.name="Owerp"
        self.data = np.delete(self.data,[1,3], axis=1)
        
        super(Power, self).__init__(*args, **kwargs)

class Mixture(RealDataset):
    
    def __init__(self, p, n_clusters, seed, *args, **kwargs):
        
        self.n_clusters=n_clusters
        self.seed = seed
        self.D = p.D
        data = np.r_[p.data, p.valid_data]
        gamma = 1.0/2.0 / np.median(pdist(data[:1000]))**2
        cluster = SpectralClustering(gamma = gamma, n_clusters=n_clusters, random_state=seed,
                            eigen_solver="arpack",
                            affinity="nearest_neighbors")
        y = cluster.fit_predict(data)
        self.ps = []
        self.props = []
        kwargs["permute"]=False
        kwargs["ntest"]  =0
        kwargs["whiten"]  =True
        kwargs["itanh"]  = False
        self.name=p.name+"_mix"

        self.test_data = p.test_data
        self.ntest    = p.test_data.shape[0]
        if hasattr(p, "has_grad"):
            self.has_grad = p.has_grad
        if hasattr(p, "grad_multiple"):
            self.grad_multiple = p.grad_multiple
        if hasattr(p, "logpdf_multiple"):
            self.logpdf_multiple = p.logpdf_multiple
        if hasattr(p, "sample"):
            self.sample = p.sample
        if hasattr(p, "noise_std"):
            self.noise_std = p.noise_std
        if hasattr(p, "idx"):
            self.idx = p.idx

        for i in range(n_clusters):
            d = data[y==i]
            p_i = ArrayDataset(d, p.name+"_%d"%i, *args, seed=seed, **kwargs)
            prop = np.sum(y==i) * 1.0 / data.shape[0]
            self.ps.append(p_i)
            self.props.append(prop)
             

class ArrayDataset(RealDataset):
    
    def __init__(self, data, name, *args, **kwargs):
        self.data = data
        self.name = name
        super(ArrayDataset, self).__init__(*args, **kwargs)

class RealToy(RealDataset):

    def __init__(self, name, D, N=10000, rotate=False, data_args={}, *args, **kwargs):
           
        self.name = name
        name = name.title() 
        d = globals()[name](D=D, **data_args)
        self.dist = d
        self.has_grad = d.has_grad
        
        np.random.seed(kwargs["seed"])
        data = d.sample(N)
    
        if rotate:

            M    = np.random.rand(D,D)*10
            M[np.triu_indices_from(M,k=1)] = 0
            M    = M - M.T
            M    = expm(M)
            
            self.M = M
        else:
            self.M = np.eye(D)
        self.data = np.dot(data, self.M.T)

        kwargs["whiten"] = False
        kwargs["itanh"] = False
        kwargs["permute"] =False

        super(RealToy, self).__init__(*args, **kwargs)

    def logpdf_multiple(self, data):
        
        return self.dist.logpdf_multiple(self.itrans(data))

    def grad_multiple(self, data):
        return self.dist.grad_multiple(self.itrans(data)).dot(self.M.T)

    def itrans(self, data):
        
        if self.itanh:
            data = inv_itanh(data, self.ptp, self.min, self.mean2)

        if self.whiten:
            data = inv_whiten(data, self.W, self.mean)
            
        data = data[:, np.argsort(self.idx)]

        data = np.dot(data, self.M)

        return data

    def score(self, x):
        return -0.5*np.sum(self.grad_multiple(x)**2,1)

def load_data(dname, noise_std=0.0, seed=1, D=None, data_args={}, **kwargs):
    dname = dname.lower()
    if dname[0:2] == "re":
        kwargs["dequantise"] = True
        p = RedWine(noise_std=noise_std, seed=seed, **kwargs)
    elif dname[0] == "w":
        kwargs["dequantise"] = True
        p = WhiteWine(noise_std=noise_std, seed=seed, **kwargs)
    elif dname[0] == 'p':
        p = Parkinsons(noise_std=noise_std, seed=seed, **kwargs)
    elif dname == 'gas':
        p = Gas(noise_std=noise_std, seed=seed, **kwargs)
    elif dname[0] == 'o':
        p = Power(noise_std=noise_std, seed=seed, **kwargs)
    elif dname[0] == 'h':
        p = HepMass(noise_std=noise_std, seed=seed, **kwargs)
    elif dname[0:2] == 'mi':
        p = MiniBoone(noise_std=noise_std, seed=seed, **kwargs)
    elif dname[0] in ['d', 'f','r', 's', 'b', 'c', 'g', 'm', 'u']:
        assert D is not None
        p = RealToy(dname.title(), D=D, noise_std=0.0, ntest=1000, seed=seed, data_args=data_args, **kwargs)
    return p


