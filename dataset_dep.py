import torch.utils.data as data
import torch,torchvision
import h5py
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
import glob, os
from PIL import Image
import numpy as np


class DatasetFromHdf5(data.Dataset):
	def __init__(self, file_path):
		super(DatasetFromHdf5, self).__init__()
		hf = h5py.File(file_path)
		self.data = hf.get("label")
	def __getitem__(self, index):
		r=random.randint(0,self.data.shape[0]-1)
		return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.data[r,:,:,:]).float()

	def __len__(self):
		return self.data.shape[0]

class DegTarDataset(Dataset):
	def __init__(self, deg_path, tar_path, pairnum):
		self.deg_path = deg_path
		self.tar_path = tar_path
		self.deg_img = sorted(glob.glob(deg_path+'*'))
		self.tar_img = sorted(glob.glob(tar_path+'*'))
		self.transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop([256,256]),torchvision.transforms.ToTensor()])
		self.pairnum = pairnum
	def __len__(self):
		return len(self.deg_img)

	def __getitem__(self, index):
		deg_img = Image.open(self.deg_img[index]).convert('RGB')
		r = random.randint(0, len(self.tar_img) - 1)
		if index<=self.pairnum:
			tar_img = Image.open(self.tar_img[index]).convert('RGB')
		else:
			tar_img = Image.open(self.tar_img[r]).convert('RGB')
		return self.transform(deg_img).float(), self.transform(tar_img).float()

class pcDataset(Dataset):
	def __init__(self, path):
		self.tar_path = path
		self.tar_pcl = glob.glob(path+'*')
	def __len__(self):
		return len(self.tar_pcl)

	def __getitem__(self, index):
		r = random.randint(0, len(self.tar_pcl) - 1)
		target = np.loadtxt(self.tar_pcl[index])
		original = np.loadtxt(self.tar_pcl[r])
		target = torch.from_numpy(target)
		original = torch.from_numpy(original)
		return target.float(), original.float()




