import numpy as np
import matplotlib.pyplot as plt
import torch
from bkregression.kernel_smoothing import KernelRegressor
from bkregression.kernels import RBF
from tqdm import tqdm
import pandas as pd
import ast

# LEVEL_1 = 1750
# LEVEL_2 = 700
# LEVEL_3 = 350
# LEVEL_4 = 175

LEVEL = 175
optimizer_file = "meta_training_175.csv"
best_parameters_file = "meta_training_175_best_parameters.csv"

df = pd.read_csv(optimizer_file)
df_best_parameters = pd.read_csv(best_parameters_file)
configs = list(df["Arm Parameterizations"])
configs = [ast.literal_eval(c) for c in configs]
configs_copy = []
for i,c in enumerate(configs):
    try:
        configs_copy.append(c[f"{i}_0"])
    except:
        pass
configs = configs_copy
best_integration_method =  list(df_best_parameters["integration_method"])[-1]
best_dt = list(df_best_parameters["dt"])[-1]
best_sensor_std = list(df_best_parameters["sensor_std"])[-1]

rk4_list = []
rk2_list = []
rk1_list = []
for c in configs:
    if c["integration_method"]=="RK4":
        rk4_list.append([c["dt"],c["sensor_std"]])
    elif c["integration_method"]=="RK2":
        rk2_list.append([c["dt"],c["sensor_std"]])
    elif c["integration_method"]=="RK1":
        rk1_list.append([c["dt"],c["sensor_std"]])
    else:
        raise NotImplementedError
rk4_list = np.array(rk4_list)
rk2_list = np.array(rk2_list)
rk1_list = np.array(rk1_list)

rk4_data = np.load("RK4_failures_10.npy")
rk2_data = np.load("RK2_failures_10.npy")
rk1_data = np.load("RK1_failures_10.npy")

dt = np.linspace(0.1,1.5,20)
s_std = np.linspace(0.1,10.0,20)
DT,S_STD = np.meshgrid(dt,s_std)
points = np.stack((DT,S_STD),axis=-1).reshape(-1,2)

dt_eval = np.linspace(0.1,1.5,200)
s_std_eval = np.linspace(0.1,10.0,200)
DT_eval,S_STD_eval = np.meshgrid(dt_eval,s_std_eval)
points_eval = np.stack((DT_eval,S_STD_eval),axis=-1).reshape(-1,2)

cost_rk4 = (25/points_eval[:,0]*(4+1/points_eval[:,1])).reshape(200,200)
cost_rk2 = (25/points_eval[:,0]*(2+1/points_eval[:,1])).reshape(200,200)
cost_rk1 = (25/points_eval[:,0]*(1+1/points_eval[:,1])).reshape(200,200)

# max_scores = []
# for _ in tqdm(range(100)):
#     scores = []
#     scales = torch.logspace(-2,-0.5,100)
#     for s in scales:
#         # print(s)
#         kernel_rk4 = RBF(s)
#         model_rk4 = KernelRegressor(torch.Tensor(points.T),torch.Tensor(np.mean(rk1_data,axis=-1).reshape(-1,)),kernel_rk4)
#         scores.append(model_rk4.k_score(10).item())
#     # print(scales[np.argmax(scores)])
#     max_scores.append(scales[np.argmax(scores)].item())
# # plt.semilogx(scales,scores)
# # plt.ylim(-10,0)
# print('stop')

points_normalized = (points-np.array([0.1,0.1]))/np.array([1.5-0.1,10.0-0.1])
points_eval_normalized = (points_eval-np.array([0.1,0.1]))/np.array([1.5-0.1,10.0-0.1])



fig,axs = plt.subplots(1,3,figsize=(15,5))
axs[0].imshow(np.mean(rk4_data,axis=-1),vmin=0,vmax=500,extent=[0.1,10.0,0.1,1.5],origin="lower",aspect="auto")
axs[0].contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL],cmap="autumn",linestyles="dashed")
if len(rk4_list)>0:
    axs[0].scatter(rk4_list[:,1],rk4_list[:,0],color=(1,0,0),alpha=0.5)
if best_integration_method == "RK4":
    axs[0].scatter(best_sensor_std,best_dt,marker='s',color=(1,1,0),s=80)
axs[0].set_title("RK4")
axs[1].imshow(np.mean(rk2_data,axis=-1),vmin=0,vmax=500,extent=[0.1,10.0,0.1,1.5],origin="lower",aspect="auto")
axs[1].contour(S_STD_eval,DT_eval,cost_rk2,levels=[LEVEL],cmap="autumn",linestyles="dashed")
if len(rk2_list)>0:
    axs[1].scatter(rk2_list[:,1],rk2_list[:,0],color=(1,0,0),alpha=0.5)
if best_integration_method == "RK2":
    axs[1].scatter(best_sensor_std,best_dt,marker='s',color=(1,1,0),s=80)
axs[1].set_title("RK2")
axs[2].imshow(np.mean(rk1_data,axis=-1),vmin=0,vmax=500,extent=[0.1,10.0,0.1,1.5],origin="lower",aspect="auto")
axs[2].contour(S_STD_eval,DT_eval,cost_rk1,levels=[LEVEL],cmap="autumn",linestyles="dashed")
if len(rk1_list)>0:
    axs[2].scatter(rk1_list[:,1],rk1_list[:,0],color=(1,0,0),alpha=0.5)
if best_integration_method == "RK1":
    axs[2].scatter(best_sensor_std,best_dt,marker='s',color=(1,1,0),s=80)
axs[2].set_title("RK1")


fig.tight_layout()
fig.show()
fig.savefig("optimization_plot_175.png",dpi=300)

print("stop")