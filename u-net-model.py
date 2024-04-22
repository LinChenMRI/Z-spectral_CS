import numpy as np
import h5py
import scipy.io
import torch.cuda

from model import *


torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


matfn = "data/data.mat"
data=h5py.File(matfn,'r')
sparse_x = data['sparse_x'][0]
sparse_x = torch.tensor(np.array(sparse_x)).to(device)
sparse_y = data['sparse_y'][0]
sparse_y = torch.tensor(np.array(sparse_y)).to(device)
dense_x = data['dense_x'][0]
dense_x = torch.tensor(np.array(dense_x)).to(device)
dense_y = data['dense_y'][0]
dense_y = torch.tensor(np.array(dense_y)).to(device)
masked_sparse_y = data['masked_sparse_y'][0]
masked_sparse_y = torch.tensor(np.array(masked_sparse_y)).to(device)
mask = data['mask'][0]
mask = torch.tensor(np.array(mask)).to(device)

net = unet().to(device)
if device.type == "cpu":
    net = torch.load("model_parameters/model_parameters.pth",map_location=torch.device('cpu'))
if device.type == "cuda":
    net = torch.load("model_parameters/model_parameters.pth")
net.eval()
with torch.no_grad():
    # predict only use mask and mask_sparse_y
    res = net(masked_sparse_y.unsqueeze(0).unsqueeze(0))
    predict = res * (1 - mask) + masked_sparse_y * mask;


predict = predict.squeeze(0).squeeze(0).cpu().numpy()
dense_y = dense_y.squeeze(0).squeeze(0).cpu().numpy()
dense_x = dense_x.squeeze(0).squeeze(0).cpu().numpy()
sparse_y = sparse_y.squeeze(0).squeeze(0).cpu().numpy()
sparse_x = sparse_x.squeeze(0).squeeze(0).cpu().numpy()

file_path = "data/demo_data.mat"
data = {
    'Z_spectra_CS_y': predict,
    'dense_y':dense_y,
    'dense_x':dense_x,
    'sparse_y':sparse_y,
    'sparse_x':sparse_x,
}

scipy.io.savemat(file_path, data)


