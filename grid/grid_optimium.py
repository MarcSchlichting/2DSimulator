import numpy as np
import matplotlib.pyplot as plt
import torch
from bkregression.kernel_smoothing import KernelRegressor
from bkregression.kernels import RBF
from tqdm import tqdm

LEVELS = [1750,700,350,175]

rk4_data = np.load("RK4_failures_10.npy")
rk2_data = np.load("RK2_failures_10.npy")
rk1_data = np.load("RK1_failures_10.npy")

rk4_mean = np.mean(rk4_data,axis=-1)
rk2_mean = np.mean(rk2_data,axis=-1)
rk1_mean = np.mean(rk1_data,axis=-1)

dt = np.linspace(0.1,1.5,20)
s_std = np.linspace(0.1,10.0,20)
DT,S_STD = np.meshgrid(dt,s_std)
points = np.stack((DT,S_STD),axis=-1).reshape(-1,2)

cost_rk4 = (25/points[:,0]*(4+1/points[:,1])).reshape(20,20)
cost_rk2 = (25/points[:,0]*(2+1/points[:,1])).reshape(20,20)
cost_rk1 = (25/points[:,0]*(1+1/points[:,1])).reshape(20,20)

cf_config = {}

for l in LEVELS:
    print(10*"-"+str(l)+10*"-")

    mask_rk4 = torch.where(torch.Tensor(cost_rk4)<=l,0,10000)
    mask_rk2 = torch.where(torch.Tensor(cost_rk2)<=l,0,10000)
    mask_rk1 = torch.where(torch.Tensor(cost_rk1)<=l,0,10000)

    best_value = np.inf
    best_dt = None
    best_im = None
    best_sensor_std = None

    rk4_min = torch.min(mask_rk4 + rk4_mean)
    rk4_argmin = torch.argmin(mask_rk4 + rk4_mean)
    if rk4_min<best_value:
        best_value = rk4_min
        best_dt = points[rk4_argmin][0]
        best_sensor_std = points[rk4_argmin][1]
        best_im = "RK4"
    print(f"RK4 - Minimum: {rk4_min} at {points[rk4_argmin]}")

    rk2_min = torch.min(mask_rk2 + rk2_mean)
    rk2_argmin = torch.argmin(mask_rk2 + rk2_mean)
    if rk2_min<best_value:
        best_value = rk2_min
        best_dt = points[rk2_argmin][0]
        best_sensor_std = points[rk2_argmin][1]
        best_im = "RK2"
    print(f"RK2 - Minimum: {rk2_min} at {points[rk2_argmin]}")

    rk1_min = torch.min(mask_rk1 + rk1_mean)
    rk1_argmin = torch.argmin(mask_rk1 + rk1_mean)
    if rk1_min<best_value:
        best_value = rk1_min
        best_dt = points[rk1_argmin][0]
        best_sensor_std = points[rk1_argmin][1]
        best_im = "RK1"
    print(f"RK1 - Minimum: {rk1_min} at {points[rk1_argmin]}")

    cf_config[l] = {"dt":best_dt,"integration_method":best_im,"sensor_std":best_sensor_std}


print(cf_config)
print("stop")

