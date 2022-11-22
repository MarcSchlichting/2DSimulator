import numpy as np
import matplotlib.pyplot as plt
import torch
from bkregression.kernel_smoothing import KernelRegressor
from bkregression.kernels import RBF
from tqdm import tqdm

rk4_data = np.load("RK4_all_10.npy")
rk2_data = np.load("RK2_all_10.npy")
rk1_data = np.load("RK1_all_10.npy")

dt = np.linspace(0.1,1.5,20)
s_std = np.linspace(0,2.0,20)
DT,S_STD = np.meshgrid(dt,s_std)
points = np.stack((DT,S_STD),axis=-1).reshape(-1,2)

dt_eval = np.linspace(0.1,1.5,200)
s_std_eval = np.linspace(0,2.0,200)
DT_eval,S_STD_eval = np.meshgrid(dt_eval,s_std_eval)
points_eval = np.stack((DT_eval,S_STD_eval),axis=-1).reshape(-1,2)

cost_rk4 = (25/points[:,0]*(4+1/points[:,1])).reshape(20,20)
cost_rk2 = (25/points[:,0]*(2+1/points[:,1])).reshape(20,20)
cost_rk1 = (25/points[:,0]*(1+1/points[:,1])).reshape(20,20)

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



kernel_rk4 = RBF(0.01)
model_rk4 = KernelRegressor(torch.Tensor(points.T),torch.Tensor(np.mean(rk4_data,axis=-1).reshape(-1,)),kernel_rk4)
model_rk4.fit(iterations=200,k=10,verbose=True)
mean_rk4,std_rk4 = model_rk4(torch.Tensor(points_eval.T))
mean_rk4 = mean_rk4.reshape(200,200)
std_rk4 = std_rk4.reshape(200,200)

kernel_rk2 = RBF(0.01)
model_rk2 = KernelRegressor(torch.Tensor(points.T),torch.Tensor(np.mean(rk2_data,axis=-1).reshape(-1,)),kernel_rk2)
model_rk2.fit(iterations=200,k=10,verbose=True)
mean_rk2,std_rk2 = model_rk2(torch.Tensor(points_eval.T))
mean_rk2 = mean_rk2.reshape(200,200)
std_rk2 = std_rk2.reshape(200,200)

kernel_rk1 = RBF(0.01)
model_rk1 = KernelRegressor(torch.Tensor(points.T),torch.Tensor(np.mean(rk1_data,axis=-1).reshape(-1,)),kernel_rk1)
model_rk1.fit(iterations=200,k=10,verbose=True)
mean_rk1,std_rk1 = model_rk1(torch.Tensor(points_eval.T))
mean_rk1 = mean_rk1.reshape(200,200)
std_rk1 = std_rk1.reshape(200,200)



fig,axs = plt.subplots(3,3,figsize=(15,10))
axs[0,0].imshow(np.mean(rk4_data,axis=-1),vmin=0,vmax=500,extent=[0,2,0.1,1.5],origin="lower")
axs[0,0].contour(S_STD,DT,cost_rk4,levels=[200],cmap="autumn",linestyles="solid")
axs[0,0].contour(S_STD,DT,cost_rk4,levels=[150],cmap="autumn",linestyles="dashdot")
axs[0,0].contour(S_STD,DT,cost_rk4,levels=[100],cmap="autumn",linestyles="dashed")
axs[0,0].set_title("RK4 - Data")
axs[0,1].imshow(np.mean(rk2_data,axis=-1),vmin=0,vmax=500,extent=[0,2,0.1,1.5],origin="lower")
axs[0,1].contour(S_STD,DT,cost_rk2,levels=[200],cmap="autumn",linestyles="solid")
axs[0,1].contour(S_STD,DT,cost_rk2,levels=[150],cmap="autumn",linestyles="dashdot")
axs[0,1].contour(S_STD,DT,cost_rk2,levels=[100],cmap="autumn",linestyles="dashed")
axs[0,1].set_title("RK2 - Data")
axs[0,2].imshow(np.mean(rk1_data,axis=-1),vmin=0,vmax=500,extent=[0,2,0.1,1.5],origin="lower")
axs[0,2].contour(S_STD,DT,cost_rk1,levels=[200],cmap="autumn",linestyles="solid")
axs[0,2].contour(S_STD,DT,cost_rk1,levels=[150],cmap="autumn",linestyles="dashdot")
axs[0,2].contour(S_STD,DT,cost_rk1,levels=[100],cmap="autumn",linestyles="dashed")
axs[0,2].set_title("RK1 - Data")

axs[1,0].imshow(mean_rk4,vmin=0,vmax=500,extent=[0,2,0.1,1.5],origin="lower")
axs[1,0].contour(S_STD,DT,cost_rk4,levels=[200],cmap="autumn",linestyles="solid")
axs[1,0].contour(S_STD,DT,cost_rk4,levels=[150],cmap="autumn",linestyles="dashdot")
axs[1,0].contour(S_STD,DT,cost_rk4,levels=[100],cmap="autumn",linestyles="dashed")
scale = model_rk4.kernel.parameters["scale"].item()
axs[1,0].set_title(f"RK4 - RBF mean ($\ell={scale:4f}$)")
axs[1,1].imshow(mean_rk2,vmin=0,vmax=500,extent=[0,2,0.1,1.5],origin="lower")
axs[1,1].contour(S_STD,DT,cost_rk2,levels=[200],cmap="autumn",linestyles="solid")
axs[1,1].contour(S_STD,DT,cost_rk2,levels=[150],cmap="autumn",linestyles="dashdot")
axs[1,1].contour(S_STD,DT,cost_rk2,levels=[100],cmap="autumn",linestyles="dashed")
scale = model_rk2.kernel.parameters["scale"].item()
axs[1,1].set_title(f"RK2 - RBF mean ($\ell={scale:4f}$)")
axs[1,2].imshow(mean_rk1,vmin=0,vmax=500,extent=[0,2,0.1,1.5],origin="lower")
axs[1,2].contour(S_STD,DT,cost_rk1,levels=[200],cmap="autumn",linestyles="solid")
axs[1,2].contour(S_STD,DT,cost_rk1,levels=[150],cmap="autumn",linestyles="dashdot")
axs[1,2].contour(S_STD,DT,cost_rk1,levels=[100],cmap="autumn",linestyles="dashed")
scale = model_rk1.kernel.parameters["scale"].item()
axs[1,2].set_title(f"RK1 - RBF mean ($\ell={scale:4f}$)")

axs[2,0].imshow(std_rk4,vmin=0,vmax=100,extent=[0,2,0.1,1.5],origin="lower",cmap="Greys_r")
axs[2,0].contour(S_STD,DT,cost_rk4,levels=[200],cmap="autumn",linestyles="solid")
axs[2,0].contour(S_STD,DT,cost_rk4,levels=[150],cmap="autumn",linestyles="dashdot")
axs[2,0].contour(S_STD,DT,cost_rk4,levels=[100],cmap="autumn",linestyles="dashed")
scale = model_rk4.kernel.parameters["scale"].item()
axs[2,0].set_title(f"RK4 - RBF std ($\ell={scale:4f}$)")
axs[2,1].imshow(std_rk2,vmin=0,vmax=100,extent=[0,2,0.1,1.5],origin="lower",cmap="Greys_r")
axs[2,1].contour(S_STD,DT,cost_rk2,levels=[200],cmap="autumn",linestyles="solid")
axs[2,1].contour(S_STD,DT,cost_rk2,levels=[150],cmap="autumn",linestyles="dashdot")
axs[2,1].contour(S_STD,DT,cost_rk2,levels=[100],cmap="autumn",linestyles="dashed")
scale = model_rk2.kernel.parameters["scale"].item()
axs[2,1].set_title(f"RK2 - RBF std ($\ell={scale:4f}$)")
axs[2,2].imshow(std_rk1,vmin=0,vmax=100,extent=[0,2,0.1,1.5],origin="lower",cmap="Greys_r")
axs[2,2].contour(S_STD,DT,cost_rk1,levels=[200],cmap="autumn",linestyles="solid")
axs[2,2].contour(S_STD,DT,cost_rk1,levels=[150],cmap="autumn",linestyles="dashdot")
axs[2,2].contour(S_STD,DT,cost_rk1,levels=[100],cmap="autumn",linestyles="dashed")
scale = model_rk1.kernel.parameters["scale"].item()
axs[2,2].set_title(f"RK1 - RBF std ($\ell={scale:4f}$)")

# axs[0].imshow(regression_std,vmin=0,vmax=70,extent=[0,2,0.1,1.5],origin="lower")

fig.tight_layout()
fig.show()
fig.savefig("grid_10.png",dpi=600)

print("stop")