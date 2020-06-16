from __future__ import print_function
from PIL import Image, ImageFilter
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import DatasetFolder,ImageFolder

import torch
from copy import deepcopy
import cv2


class Imagenet32(VisionDataset):
    """
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,cuda=False, sz=32):

        super(Imagenet32, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.base_folder = root
        self.train = train  # training set or test set
        self.cuda = cuda

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i in range(1,11):
            file_name = 'train_data_batch_'+str(i)
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.targets = [t-1 for t in self.targets]
        self.data = np.vstack(self.data).reshape(-1, 3, sz, sz)
        if self.cuda:
            import torch
            self.data = torch.FloatTensor(self.data).half().cuda()#type(torch.cuda.HalfTensor)
        else:
            self.data = self.data.transpose((0,2,3,1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.cuda:
            img = self.transform(img)
            return img,target

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)









class Imagenet64(VisionDataset):
    """
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,cuda=False, sz=32):

        super(Imagenet64, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.base_folder = root
        self.train = train  # training set or test set
        self.cuda = cuda

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i in range(1,11):
            file_name = 'train_data_batch_'+str(i)
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.targets = [t-1 for t in self.targets]
        self.data = np.vstack(self.data).reshape(-1, 3, sz, sz)
        if self.cuda:
            import torch
            self.data = torch.FloatTensor(self.data).half().cuda()#type(torch.cuda.HalfTensor)
        else:
            self.data = self.data.transpose((0,2,3,1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.cuda:
            img = self.transform(img)
            return img,target

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)






IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

SELECTED_CALSSES = {
    'dish': [806, 830, 900, 948, 771, 844, 885, 993],
    'bovid':  [165, 81, 52, 9, 162, 108],
    'aqua_bird': [420, 418, 419,421, 439, 438],
    'edge_tool': [372, 375, 378, 379, 380, 377],
    'snake' : [490, 477,478,479,480,481,482,487],
    'fish' : [444, 442, 443, 445, 446, 447, 448],
    'precussion': [335, 336, 337, 338, 339, 340],
    'stinged': [341, 342, 343, 344,345, 346],
    'car' : [265, 266, 267, 268, 269, 272, 273],
    'boat' : [235, 236, 237, 238, 239, 240]
}




# SELECTED_CALSSES = {
#     'dish': [806],
#     'boat' : [235]
# }


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class MiniImagenet(DatasetFolder):
    """
    """

    def __init__(self, root,mat_fname,selected_classes=None, transform=None,augment_transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        self.selected_classes = selected_classes
        self.mat_fname = mat_fname
        super(MiniImagenet, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.data = None
        self.targets = None
        self.augment_transform = augment_transform

    def _find_classes(self, dir):
        """
        """

        if sys.version_info >= (3, 5):
           # Faster and available in Python 3.5 and above
           classes_dir = [d.name for d in os.scandir(dir) if d.is_dir() ]
        else:
           classes_dir = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) ]
        classes_dir.sort()



        class_to_idx = select_classes(self.mat_fname, self.selected_classes)

        classes = list(class_to_idx.keys())

        count_non_in_dir = 0
        for c in classes:
            if c not in classes_dir:
                count_non_in_dir  +=1
        if count_non_in_dir>0:
            raise NotImplementedError

        return classes, class_to_idx

    def load_data(self,dataloader,data_dir, force_recompute):
        if self.data is None:
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            data_name =os.path.join(data_dir, f'data.t7')
            targets_name =os.path.join(data_dir, f'targets.t7')
            count  = 0
            if (not os.path.exists(data_name)) or (not os.path.exists(targets_name)) or force_recompute:

                data = []
                targets = []
                for idx ,(inputs, target) in enumerate(dataloader):
                    # count += 1
                    # if count >10:
                    #     break
                    tmp_inputs = deepcopy(inputs.cpu().numpy())
                    tmp_target =  deepcopy(target.cpu().numpy())
                    data.append(tmp_inputs)
                    targets.append(tmp_target)
                    del inputs
                    del target
                    #data= [inputs.numpy()]
                    #print(idx)
                data = np.vstack(data)
                targets = np.concatenate(targets)
                data = data.transpose((0,2,3,1))
            else:
                data = torch.load(torch_data)
                targets = torch.load(torch_targets)
                targets = targets.numpy()
                data = data.numpy()
            self.data = data
            self.targets= targets


def select_classes(mat_fname, selected_classes = SELECTED_CALSSES):
    import scipy.io as sio
    mat_contents = sio.loadmat(mat_fname)

    classes = {}

    for i, key_values in enumerate(selected_classes.items()):
        key, value = key_values
        for idx in value:
            WNID = mat_contents['synsets'][idx-1][0][1][0]
            classes[WNID] = i

    return classes


# color distortion from https://arxiv.org/pdf/2002.05709.pdf

from torchvision import transforms

def get_color_distortion(s=0.1):
    # s is the strength of color distortion.
    
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter,rnd_gray])
    return color_distort

def gaussian_blur(img):
    size = int(img.height/10.)
    image_blur = cv2.GaussianBlur(img,(size,size),0.1)
    new_image = image_blur
    return new_image


def get_data_augmentation(spatial_size,normalize,color_dist=1.,g_blur=0.,affine=False):
    blur = transforms.Lambda(gaussian_blur)
    rnd_blur = transforms.RandomApply([blur], p=0.5)
    color_distort = get_color_distortion(s=color_dist)
    affine = transforms.RandomAffine(10, scale=(0.8,1.2),shear=[-0.1,0.1,-0.1,0.1])
    augmentation_transforms = [
    transforms.ToPILImage(),
    transforms.RandomCrop(spatial_size),
    transforms.RandomHorizontalFlip()]
    if affine:
        augmentation_transforms.append(affine)
    if color_dist>0.:
        augmentation_transforms.append(color_distort)
    if g_blur>0.:
        augmentation_transforms.append(rnd_blur)

    augmentation_transforms.append(transforms.ToTensor())
    augmentation_transforms.append(normalize)
    
    return transforms.Compose(augmentation_transforms)






