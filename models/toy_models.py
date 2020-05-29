from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.utils import spectral_norm as sn_official
spectral_norm = sn_official
import samplers
import os
class Generator(nn.Module):
    def __init__(self, dim=3,  device ='cuda'):
        super(Generator, self).__init__()
        W = torch.eye(dim).float().to(device)
        U = torch.zeros([dim,dim]).float().to(device)
        U = torch.from_numpy( np.array([[1.,0,0],[0,1.,0],[0,0,1.]]) ).float().to(device)
        self.W = nn.Linear(dim,dim, bias=False)
        self.U = nn.Linear(dim,dim, bias=False)
        
        self.W.weight.data = W
        self.U.weight.data = U

        self.W.weight.requires_grad = False

        #self.U = U
        

    def forward(self, latent):
        # latent is a gaussian in 3D
        # normalize to get a uniform on the sphere
        
        Z = latent/torch.norm(latent, dim=1).unsqueeze(-1)
        U = self.W(Z) +self.U(Z**4)
        R = torch.sum(U**2, dim=1).unsqueeze(-1)
        
        return R*Z

class Discriminator(nn.Module):
    def __init__(self,dim, device='cuda', sn = True):

        super(Discriminator, self).__init__()
        W_1 =torch.randn([dim, dim]).float().to(device)
        W_2 =torch.randn([dim, dim]).float().to(device)
        W_3 =torch.randn([dim, dim]).float().to(device)
        W_4 =torch.randn([dim, dim]).float().to(device)
        U = torch.ones([dim]).float().to(device)/np.sqrt(dim)

        self.W_1 = nn.Linear(dim,dim, bias=False)
        #self.W_2 = nn.Linear(dim,dim, bias=False)
        #self.W_3 = nn.Linear(dim,dim, bias=False)
        #self.W_4 = nn.Linear(dim,dim, bias=False)
        self.U = nn.Linear(dim,1, bias=False)
        
        self.W_1.weight.data = W_1
        #self.W_2.weight.data = W_2
        #self.W_3.weight.data = W_3
        #self.W_4.weight.data = W_4
        self.U.weight.data = U


        self.leak = 0.1
        self.sn = sn
        if self.sn:
            self.main = nn.Sequential( spectral_norm(self.W_1), 
                nn.LeakyReLU(self.leak),  
                #spectral_norm(self.W_2),
                #nn.LeakyReLU(self.leak),
                #spectral_norm(self.W_3),
                #nn.LeakyReLU(self.leak), 
                #spectral_norm(self.W_4),
                #nn.LeakyReLU(self.leak),  
                spectral_norm(self.U)
                )
        else:
            self.main = nn.Sequential( self.W_1, 
                nn.LeakyReLU(self.leak),  
                #self.W_2,
                #nn.LeakyReLU(self.leak), 
                #self.W_3,
                #nn.LeakyReLU(self.leak), 
                #self.W_4,
                #nn.LeakyReLU(self.leak), 
                self.U
                )

    def forward(self, data):

        return self.main(data)

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, N_samples, dtype, device, b_size, root):
        self.total_size = N_samples
        self.cur_index = b_size
        self.b_size =  b_size
        self.device = device
        self.N_samples = N_samples
        self.dtype=dtype
        D=3
        self.base = Generator(dim=D).to(self.device)
        self.base.W.weight.data = torch.eye(3).float().to(self.device)
        self.base.U.weight.data = torch.from_numpy( np.array([[1.,0,0],[0,1.,0],[0,0,1.]]) ).float().to(self.device)

        self.energy = Discriminator(dim=D, sn=False).to(device)
        W_1 = np.array([[0.1,-1.,-1.],[-1.,0.1,-1.],[-1.,-1.,0.1]])
        W_2 = np.array([[-0.1,-1.,1.],[-1.,-0.1,1.],[1.,-1.,-0.1]])
        W_3 = np.array([[-0.1,-1.,1.],[-1.,-0.1,1.],[1.,-1.,-0.1]])
        W_4 = np.array([[0.1,-1.,-1.],[-1.,0.1,-1.],[-1.,-1.,0.1]])
        U = np.array([1.,1.,1.])/np.sqrt(3)
        _,S_1,_ = np.linalg.svd(W_1)
        _,S_2,_ = np.linalg.svd(W_2)
        _,S_3,_ = np.linalg.svd(W_3)
        _,S_4,_ = np.linalg.svd(W_4)
        W_1  = W_1/S_1[0]
        W_2 = W_2/S_2[0]
        W_3  = W_3/S_3[0]
        W_4 = W_4/S_4[0]

        self.energy.W_1.weight.data = torch.from_numpy(W_1).float().to(self.device)
        #self.energy.W_2.weight.data = torch.from_numpy(W_2).float().to(self.device)
        #self.energy.W_3.weight.data = torch.from_numpy(W_3).float().to(self.device)
        #self.energy.W_4.weight.data = torch.from_numpy(W_4).float().to(self.device)

        
        self.energy.U.weight.data = torch.from_numpy( U ).float().to(self.device)
        
        self.source = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros([D] ).to(self.device), torch.eye(D).to(self.device))
        self.latent_potential = samplers.Latent_potential(self.base,self.energy, self.source, 1.)
        self.latent_sampler = samplers.MALA(self.latent_potential, T=1000, gamma=1e-3) 
        
        file_name = 'toy_data'
        data_path = os.path.join(root, file_name)
        if not os.path.exists(data_path+'.npz'):
            data, latents, noise  = self.make_data()
            np.savez(data_path, data=data.cpu().numpy(),
                    latents=latents.cpu().numpy(), noise=noise.cpu().numpy())
        else:
            dataset =  np.load(data_path+'.npz')
            self.data = torch.from_numpy(dataset['data'])
            self.latents = torch.from_numpy(dataset['latents'])
            self.noise = torch.from_numpy(dataset['noise'])
        self.counter = 0
    def make_data(self):

        self.noise =torch.cat([ self.source.sample([self.b_size]).cpu() for b in range(int( self.N_samples/self.b_size)+1)], dim=0)
        self.latents = self.sample_latents(self.noise , T = 100)
        self.data =  self.sample_data(self.latents)
        return self.data, self.latents, self.noise

    def __len__(self):
        return self.total_size
    def sample_latents(self,priors, T, with_acceptance = False):
        #return priors
        posteriors = []
        avg_acceptences = []
        for b, prior in enumerate(priors.split(self.b_size, dim=0)):
            prior = prior.clone().to(self.device)
            posterior,avg_acceptence = self.latent_sampler.sample(prior,sample_chain=False,T=T)            
            posteriors.append(posterior)
            avg_acceptences.append(avg_acceptence)

        posteriors = torch.cat(posteriors, axis=0)
        avg_acceptences = np.mean(np.array(avg_acceptences), axis=0)

        if with_acceptance:
            return posteriors, avg_acceptences
        else:
            return posteriors
    def sample_data(self, latents,  to_cpu = True, as_list=False):
        self.base.eval()
        self.energy.eval()
        images = []
        for latent in latents.split(self.b_size, dim=0):
            with torch.no_grad():
                img = self.base(latent.to(self.device))
            if to_cpu:
                img = img.cpu()
            images.append(img)
        if as_list:
            return images
        else:
            return torch.cat(images, dim=0)

    def __getitem__(self,index):
        self.counter +=1
        if np.mod(self.counter, 100.*self.N_samples ) ==0:
            print('sampling data')
            self.latents = self.sample_latents(self.latents , T = 10)
            self.data =  self.sample_data(self.latents)

        return self.data[index],self.data[index] 
