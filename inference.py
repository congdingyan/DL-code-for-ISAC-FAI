import numpy as np
import h5py
import torch
from model import *
import matplotlib.pyplot as plt
import scipy.io as scio


#Download Data
feature = h5py.File('test_data.mat')
H_data = feature['H_data'][:]
H_data = np.transpose(H_data)

#transform to tensor
H_data_tensor = torch.from_numpy(H_data)
H_data_tensor = H_data_tensor.to(torch.float32)

#Normalized
H_data_tensor=(H_data_tensor+3.5000)/7

#Test Set
test_data_x =H_data_tensor[0:500]

#model
net = torch.load('mat.pkl')

#inference
test_output = net(test_data_x[:500])

#Normalized
test_output = test_output/2 - 0.2500

#transform to numpy
test_output=test_output.detach().numpy()
scio.savemat('dl.mat', {'dl_data':test_output})

