import numpy as np



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
            print('Filling the buffer ')
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