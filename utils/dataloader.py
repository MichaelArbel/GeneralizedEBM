# Adapted from https://github.com/kevin-w-li/deep-kexpfam/blob/master/Datasets.py


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
import h5py as h5
from scipy.stats import truncnorm as tnorm
from scipy.linalg import expm
#from autograd import elementwise_grad

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans, SpectralClustering
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torch.autograd import Variable
import torch.functional as F




class PrepareUCIData(Dataset):

    def __init__(self,X):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X).float()
            self.y = torch.zeros([X.shape[0]])
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
