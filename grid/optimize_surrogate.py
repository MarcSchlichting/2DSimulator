import numpy as np
import matplotlib.pyplot as plt
import torch
from bkregression.kernel_smoothing import KernelRegressor
from bkregression.kernels import RBF
from tqdm import tqdm

LEVELS = [1750,700,350,175]

rk4_data = np.load("RK4_all_50.npy")
rk2_data = np.load("RK2_all_50.npy")
rk1_data = np.load("RK1_all_50.npy")

dt = np.linspace(0.1,1.5,20)
s_std = np.linspace(0.1,10.0,20)
DT,S_STD = np.meshgrid(dt,s_std)
points = np.stack((DT,S_STD),axis=-1).reshape(-1,2)

dt_eval = np.linspace(0.1,1.5,1000)
s_std_eval = np.linspace(0.1,10.0,1000)
DT_eval,S_STD_eval = np.meshgrid(dt_eval,s_std_eval)
points_eval = np.stack((DT_eval,S_STD_eval),axis=-1).reshape(-1,2)

cost_rk4 = (25/points_eval[:,0]*(4+1/points_eval[:,1])).reshape(1000,1000)
cost_rk2 = (25/points_eval[:,0]*(2+1/points_eval[:,1])).reshape(1000,1000)
cost_rk1 = (25/points_eval[:,0]*(1+1/points_eval[:,1])).reshape(1000,1000)



points_normalized = (points-np.array([0.1,0.1]))/np.array([1.5-0.1,10.0-0.1])
points_eval_normalized = (points_eval-np.array([0.1,0.1]))/np.array([1.5-0.1,10.0-0.1])


kernel_rk4 = RBF(0.01)
model_rk4 = KernelRegressor(torch.Tensor(points_normalized.T),torch.Tensor(np.mean(rk4_data,axis=-1).reshape(-1,)),kernel_rk4)
model_rk4.fit(iterations=200,k=10,verbose=True)
mean_rk4,std_rk4 = model_rk4(torch.Tensor(points_eval_normalized.T))
mean_rk4 = mean_rk4.reshape(1000,1000)
std_rk4 = std_rk4.reshape(1000,1000)

kernel_rk2 = RBF(0.01)
model_rk2 = KernelRegressor(torch.Tensor(points_normalized.T),torch.Tensor(np.mean(rk2_data,axis=-1).reshape(-1,)),kernel_rk2)
model_rk2.fit(iterations=200,k=10,verbose=True)
mean_rk2,std_rk2 = model_rk2(torch.Tensor(points_eval_normalized.T))
mean_rk2 = mean_rk2.reshape(1000,1000)
std_rk2 = std_rk2.reshape(1000,1000)

kernel_rk1 = RBF(0.01)
model_rk1 = KernelRegressor(torch.Tensor(points_normalized.T),torch.Tensor(np.mean(rk1_data,axis=-1).reshape(-1,)),kernel_rk1)
model_rk1.fit(iterations=200,k=10,verbose=True)
mean_rk1,std_rk1 = model_rk1(torch.Tensor(points_eval_normalized.T))
mean_rk1 = mean_rk1.reshape(1000,1000)
std_rk1 = std_rk1.reshape(1000,1000)

for l in LEVELS:
    print(10*"-"+str(l)+10*"-")

    mask_rk4 = torch.where(torch.Tensor(cost_rk4)<=l,0,10000)
    mask_rk2 = torch.where(torch.Tensor(cost_rk2)<=l,0,10000)
    mask_rk1 = torch.where(torch.Tensor(cost_rk1)<=l,0,10000)

    rk4_min = torch.min(mask_rk4 + mean_rk4)
    rk4_argmin = torch.argmin(mask_rk4 + mean_rk4)
    print(f"RK4 - Minimum: {rk4_min} at {points_eval[rk4_argmin]}")

    rk2_min = torch.min(mask_rk2 + mean_rk2)
    rk2_argmin = torch.argmin(mask_rk2 + mean_rk2)
    print(f"RK2 - Minimum: {rk2_min} at {points_eval[rk2_argmin]}")

    rk1_min = torch.min(mask_rk1 + mean_rk1)
    rk1_argmin = torch.argmin(mask_rk1 + mean_rk1)
    print(f"RK1 - Minimum: {rk1_min} at {points_eval[rk1_argmin]}")

print("stop")