'''
Dataloader for 2D regression task on POCUS vs Cart-based images
'''

import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image

from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale

import numpy as np
import scipy.io as sio

import glob


def get_dataset_paths(dataset="US", data_option="C"):
    ''' get a list of filepaths pointing to dataset .mat files
    dataset: choice of dataset "US" or "celebA"
    data_option == 'C' or 'P' for cart-based or POCUS
    '''
    if dataset == "US":
        s1 = r"D:\Datasets\vq-data-copy\0\mat_data\*-"
        s2 = "-*.mat"
        query = s1 + data_option + s2
    elif dataset == "celebA":
        query = r"D:\Datasets\celebA\img_align_celeba\*.jpg"
    return glob.glob(query)

def get_2d_dataloader(sidelength, bsize, shuff, img_paths, dim_option="2D"):
    
    dset = ImageFitting(sidelength, img_paths, dim_option)
    dataloader = DataLoader(dset, batch_size=bsize, shuffle=shuff,
                            pin_memory=True, num_workers=0)
    return dataloader
    
def get_3d_dataloader(bsize, shuff, vol_path):
    '''

    Parameters
    ----------
    bsize : int
        batch size
    shuff : bool
        true/false for shuffle dataloader
    vol_path : str
        path of the 3d volume

    Returns
    -------
    dataloader : Dataloader object

    '''
    dset = VolumeFitting(vol_path)
    dataloader = DataLoader(dset, batch_size=bsize, shuffle=shuff, 
                            pin_memory=True, num_workers=0)
    return dataloader

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class ImageFitting(Dataset):
    '''Creates a dataset of (pixel, intensity) pairs for 2D image regression
        reading from .mat file
    sidelength: int
    dim_option: str for emitting 2D/3Dslice/3Drandom meshgrid that
        maps the image to meshgrid coordinates
    image_paths: list of paths towards images in dataset
    '''
        
    def __init__(self, sidelength, image_paths, dim_option="2D"):
        super().__init__()
        self.sidelength = sidelength
        self.paths_list = image_paths
        self.transform = Compose([
            Resize((self.sidelength,self.sidelength)),
            Grayscale(),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        self.dim_option = dim_option
    
    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, idx):    
        img_path = self.paths_list[idx]
        # load the .mat file
        if ".mat" in str(img_path):
            img_arr = read_mat_image(img_path, 'random')
            orig_img = Image.fromarray(img_arr)   
        elif ".jpg" in str(img_path):
            orig_img = Image.open(img_path)
        
        transformed_img = self.transform(orig_img)
        pixels = transformed_img.permute(1, 2, 0).view(-1, 1)
        coords = get_mgrid(self.sidelength, 2)
        
        if "3D" in self.dim_option:
            x_coord = 2*(idx/self.sidelength)-1 # uniformly placed between [-1,1]
            coords_x = torch.full((self.sidelength**2,1), x_coord)
            coords = torch.cat((coords_x, coords), dim=1)
            if self.dim_option=="3D_random": # shuffle x, y, z axes
                coords = coords[:, torch.randperm(3)]
        
        return coords, pixels

class VolumeFitting(Dataset):
    '''Creates a dataset of (pixel, intensity) pairs for
    reading single 3D volume from HxWxL .npy file
    volume_path: path of 3D volume
    '''
    def __init__(self, volume_path):
        super().__init__()
        self.transform = Compose([
            #Resize((self.sidelength,self.sidelength)),
            #Grayscale(),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        data = np.load(volume_path)
        assert (data.shape[0] == data.shape[1] == data.shape[2])
        self.sidelength = data.shape[0]
        self.data = self.transform(data)
    
    def __len__(self):
        return self.sidelength

    def __getitem__(self, idx):    
        vol_slice = self.data[idx]
        pixels = vol_slice.view(-1, 1)
        coords_yz = get_mgrid(self.sidelength, 2)
        x_coord = 2*(idx/self.sidelength)-1 # uniformly placed between [-1,1]
        coords_x = torch.full((self.sidelength**2,1), x_coord)
        coords = torch.cat((coords_x, coords_yz), dim=1)
        
        return coords, pixels

def read_mat_image(img_path, frame_option='random'):
    '''
    read image as HxW uint8 from the cine mat file at img_path

    Parameters
    ----------
    img_path : str
        path to the .mat file containing the medical image
    frame_option : str, optional
        Which frame to extract from the cine. The default is 'random'.

    Returns
    -------
    d : HxW numpy array of image

    '''
    try:
        matfile = sio.loadmat(img_path, verify_compressed_data_integrity=False)
    # except ValueError:
    #     print(fa)
    #     raise ValueError()
    except TypeError:
        print(img_path)
        raise TypeError()

    d = matfile['Patient']['DicomImage'][0][0]

    # if 'ImageCroppingMask' in matfile['Patient'].dtype.names:
        # mask = matfile['Patient']['ImageCroppingMask'][0][0]
        # # d = d*np.expand_dims(mask, axis=2)

    if frame_option == 'start':
        d = d[:, :, 0]
    elif frame_option == 'end':
        d = d[:, :, -1]
    elif frame_option == 'random':
        r = np.random.randint(0, d.shape[2])
        d = d[:, :, r]

    return d    
# Testing functionality of ImageFitting
# img_paths = get_dataset_paths(dataset="US", data_option="C")
# dl = get_2d_dataloader(208, 4, True, img_paths, "3D_random")
# p = VolumeFitting('../3dus_example.npy')
# dl = DataLoader(p, batch_size=4, shuffle=False, pin_memory=True, num_workers=0)
# model_input, ground_truth = next(iter(dl))
# import matplotlib.pyplot as plt
# plt.imshow(ground_truth[11].cpu().view(208,208).numpy())