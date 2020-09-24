import numpy as np
import torch
import scipy
_eps = 1.0e-5

class FIDScheduler(object):
    def __init__(self,args):
        self.freq_fid = 2000 # args.freq_fid
        self.oldest_fid_iter = 20000 # args.oldest_fid_iter
        self.num_old_fids = int(self.oldest_fid_iter/self.freq_fid) +1
        self.curent_cursor = -self.num_old_fids
        self.old_fids = np.zeros([self.num_old_fids])
        self.trainer = None
        self.num_failures = 0
        self.max_failures = 3 #args.max_failures
    def init_trainer(self, trainer):
        self.trainer = trainer

    def step(self, fid):
        if self.curent_cursor < 0:
            print('Filling the buffer: curent_cursor'+ str(self.curent_cursor))
            self.old_fids[self.num_old_fids + self.curent_cursor] = fid
            self.curent_cursor += 1
        else:
            print(f'old_fids')
            print(self.old_fids)
            self.old_fids[self.curent_cursor] = fid
            self.curent_cursor = np.mod(self.curent_cursor+1, self.num_old_fids)
            old_fid = self.old_fids[self.curent_cursor]
            print('new_fids')
            print(self.old_fids)

            if old_fid < fid:
                print(' incresing counter  ')
                self.num_failures += 1 
            else:
                print('resetting counter')
                self.num_failures = 0
            if self.num_failures==self.max_failures:
                print(" reducing step-size ")
                self.num_failures = 0
                self.trainer.scheduler_d.step()
                self.trainer.scheduler_g.step()



class MMDScheduler(object):
    def __init__(self,args, device):
        self.freq_fid = args.freq_fid
        self.oldest_fid_iter = args.oldest_fid_iter
        self.num_old_fids = int(self.oldest_fid_iter/self.freq_fid) +1
        self.curent_cursor = -self.num_old_fids
        self.bs = 2048
        self.old_scores = self.num_old_fids*[None]
        self.old_fids = np.zeros([self.num_old_fids])
        self.trainer = None
        self.device= device
        self.num_failures = 0
        self.max_failures = 3 #args.max_failures
        self.restart = 0
    def init_trainer(self, trainer):
        self.trainer = trainer

    def step(self,fid, score_true,score_fake):
        bs = min(self.bs, score_true.shape[0],score_fake.shape[0])
        act_true = score_true[:bs]
        act_fake = score_fake[:bs]
        if self.curent_cursor < 0:
            print('Filling the buffer: curent_cursor '+ str(self.curent_cursor))
            Y_related_sums = diff_polynomial_mmd2_and_ratio_with_saving(act_true.to(self.device), act_fake.to(self.device), None, device = self.device)
            self.old_scores[self.num_old_fids + self.curent_cursor] = Y_related_sums
            self.old_fids[self.num_old_fids + self.curent_cursor] = fid
            self.curent_cursor += 1
        else:
            if self.restart<0:
                print('Re-Filling the buffer: curent_cursor '+ str(self.curent_cursor))
                Y_related_sums = diff_polynomial_mmd2_and_ratio_with_saving(act_true, act_fake, None,device = self.device)
                self.old_scores[self.curent_cursor] = Y_related_sums
                self.old_fids[self.curent_cursor] = fid
                self.curent_cursor = np.mod(self.curent_cursor+1, self.num_old_fids)
                self.restart +=1
            else:
                saved_Z = self.old_scores[self.curent_cursor]
                mmd2_diff, test_stat, Y_related_sums = diff_polynomial_mmd2_and_ratio_with_saving(act_true, act_fake, saved_Z, device=self.device)
                p_val = scipy.stats.norm.cdf(test_stat)
                self.old_scores[self.curent_cursor] = Y_related_sums
                self.old_fids[self.curent_cursor] = fid
                self.curent_cursor = np.mod(self.curent_cursor+1, self.num_old_fids)
                print("3-sample test stat = %.1f" % test_stat)
                print("3-sample p-value = %.1f" % p_val)
                if p_val>.1:
                    self.num_failures += 1
                    print(' increasing counter to %d ', self.num_failures)
                    if self.num_failures>=self.max_failures:
                        self.num_failures = 0
                        self.trainer.scheduler_d.step()
                        self.trainer.scheduler_g.step()
                        self.restart = -self.max_failures
                        print("failure to improve after %d tests", self.max_failures)
                        print(" reducing lr to:  lr energy at %f  and lr gen at %f ",self.trainer.optim_d.param_groups[0]['lr'],self.trainer.optim_g.param_groups[0]['lr'])
                    else:
                        print(" No improvement in last %d, keeping lr energy at %f  and lr gen at %f ",self.num_failures,self.trainer.optim_d.param_groups[0]['lr'],self.trainer.optim_g.param_groups[0]['lr'])
                else:
                    print(" Keeping lr energy at %f  and lr gen at %f ",self.trainer.optim_d.param_groups[0]['lr'],self.trainer.optim_g.param_groups[0]['lr'])
                    self.num_failures = 0
        print("FID scores: " + str(self.old_fids))

def diff_polynomial_mmd2_and_ratio_with_saving(X, Y, saved_sums_for_Z, device='cuda'):
    dim = float(X.shape[1])
    X = X.to(device)
    Y = Y.to(device)
    # TODO: could definitely do this faster
    torch.einsum('ni,mi->nm',X,Y)
    K_XY = (torch.einsum('ni,mi->nm',X,Y) / dim + 1) ** 3
    K_YY = (torch.einsum('ni,mi->nm',Y,Y) / dim + 1) ** 3
    #K_XY = (np.dot(X, Y.transpose()) / dim + 1) ** 3
    #K_YY = (np.dot(Y, Y.transpose()) / dim + 1) ** 3
    m = float(K_YY.shape[0])

    Y_related_sums = _get_sums(K_XY, K_YY)

    if saved_sums_for_Z is None:
        return tuple([el.cpu() for el in Y_related_sums])
    saved_sums_for_Z = tuple([el.to(device) for el in saved_sums_for_Z])
    mmd2_diff, ratio = _diff_mmd2_and_ratio_from_sums(Y_related_sums, saved_sums_for_Z, m)

    return mmd2_diff, ratio, tuple([el.cpu() for el in Y_related_sums])

def _get_sums(K_XY, K_YY, const_diagonal=False):
    m = float(K_YY.shape[0])  # Assumes X, Y, Z are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to explicitly form them
    if const_diagonal is not False:
        const_diagonal = float(const_diagonal)
        diag_Y = const_diagonal
        sum_diag2_Y = m * const_diagonal**2
    else:
        diag_Y = torch.diag(K_YY)

        sum_diag2_Y = torch.sum(diag_Y**2)

    Kt_YY_sums = torch.sum(K_YY, dim=1) - diag_Y

    K_XY_sums_0 = torch.sum(K_XY, dim=0)
    K_XY_sums_1 = torch.sum(K_XY, dim=1)

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    K_XY_2_sum = (K_XY ** 2).sum()

    return Kt_YY_sums, Kt_YY_2_sum, K_XY_sums_0, K_XY_sums_1, K_XY_2_sum

def _diff_mmd2_and_ratio_from_sums(Y_related_sums, Z_related_sums, m, const_diagonal=False):
    Kt_YY_sums, Kt_YY_2_sum, K_XY_sums_0, K_XY_sums_1, K_XY_2_sum = Y_related_sums
    Kt_ZZ_sums, Kt_ZZ_2_sum, K_XZ_sums_0, K_XZ_sums_1, K_XZ_2_sum = Z_related_sums

    Kt_YY_sum = Kt_YY_sums.sum()
    Kt_ZZ_sum = Kt_ZZ_sums.sum()

    K_XY_sum = K_XY_sums_0.sum()
    K_XZ_sum = K_XZ_sums_0.sum()

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...

    ### Estimators for the various terms involved
    muY_muY = Kt_YY_sum / (m * (m-1))
    muZ_muZ = Kt_ZZ_sum / (m * (m-1))

    muX_muY = K_XY_sum / (m * m)
    muX_muZ = K_XZ_sum / (m * m)

    E_y_muY_sq = (torch.dot(Kt_YY_sums, Kt_YY_sums) - Kt_YY_2_sum) / (m*(m-1)*(m-2))
    E_z_muZ_sq = (torch.dot(Kt_ZZ_sums, Kt_ZZ_sums) - Kt_ZZ_2_sum) / (m*(m-1)*(m-2))

    E_x_muY_sq = (torch.dot(K_XY_sums_1, K_XY_sums_1) - K_XY_2_sum) / (m*m*(m-1))
    E_x_muZ_sq = (torch.dot(K_XZ_sums_1, K_XZ_sums_1) - K_XZ_2_sum) / (m*m*(m-1))

    E_y_muX_sq = (torch.dot(K_XY_sums_0, K_XY_sums_0) - K_XY_2_sum) / (m*m*(m-1))
    E_z_muX_sq = (torch.dot(K_XZ_sums_0, K_XZ_sums_0) - K_XZ_2_sum) / (m*m*(m-1))

    E_y_muY_y_muX = torch.dot(Kt_YY_sums, K_XY_sums_0) / (m*m*(m-1))
    E_z_muZ_z_muX = torch.dot(Kt_ZZ_sums, K_XZ_sums_0) / (m*m*(m-1))

    E_x_muY_x_muZ = torch.dot(K_XY_sums_1, K_XZ_sums_1) / (m*m*m)

    E_kyy2 = Kt_YY_2_sum / (m * (m-1))
    E_kzz2 = Kt_ZZ_2_sum / (m * (m-1))

    E_kxy2 = K_XY_2_sum / (m * m)
    E_kxz2 = K_XZ_2_sum / (m * m)

    ### Combine into overall estimators
    mmd2_diff = muY_muY - 2 * muX_muY - muZ_muZ + 2 * muX_muZ

    first_order = 4 * (m-2) / (m * (m-1)) * (
        E_y_muY_sq - muY_muY**2
        + E_x_muY_sq - muX_muY**2
        + E_y_muX_sq - muX_muY**2
        + E_z_muZ_sq - muZ_muZ**2
        + E_x_muZ_sq - muX_muZ**2
        + E_z_muX_sq - muX_muZ**2
        - 2 * E_y_muY_y_muX + 2 * muY_muY * muX_muY
        - 2 * E_x_muY_x_muZ + 2 * muX_muY * muX_muZ
        - 2 * E_z_muZ_z_muX + 2 * muZ_muZ * muX_muZ
    )
    second_order = 2 / (m * (m-1)) * (
        E_kyy2 - muY_muY**2
        + 2 * E_kxy2 - 2 * muX_muY**2
        + E_kzz2 - muZ_muZ**2
        + 2 * E_kxz2 - 2 * muX_muZ**2
        - 4 * E_y_muY_y_muX + 4 * muY_muY * muX_muY
        - 4 * E_x_muY_x_muZ + 4 * muX_muY * muX_muZ
        - 4 * E_z_muZ_z_muX + 4 * muZ_muZ * muX_muZ
    )
    var_est = first_order + second_order

    ratio = mmd2_diff.item() / np.sqrt(max(var_est.item(), _eps))
    return mmd2_diff.item(), ratio