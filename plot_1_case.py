import numpy as np
import matplotlib.pyplot as plt
import torch
from bkregression.kernel_smoothing import KernelRegressor
from bkregression.kernels import RBF
from tqdm import tqdm

LEVEL_1 = 1750
LEVEL_2 = 700
LEVEL_3 = 350
LEVEL_4 = 175

rk4_data = np.load("RK4_all_10.npy")
rk2_data = np.load("RK2_all_10.npy")
rk1_data = np.load("RK1_all_10.npy")

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


points_normalized = (points-np.array([0.1,0.1]))/np.array([1.5-0.1,10.0-0.1])
points_eval_normalized = (points_eval-np.array([0.1,0.1]))/np.array([1.5-0.1,10.0-0.1])






fig,axs = plt.subplots(1,1,figsize=(5,3))
axs.imshow(np.mean(rk1_data,axis=-1),vmin=0,vmax=500,extent=[0.1,10.0,0.1,1.5],origin="lower",aspect="auto")
cn = axs.contour(S_STD_eval,DT_eval,cost_rk1,levels=[LEVEL_4],cmap="autumn",linestyles="solid")
contours = []

for cc in cn.collections:
    paths = []
    # for each separate section of the contour line
    for pp in cc.get_paths():
        xy = []
        # for each segment of that section
        for vv in pp.iter_segments():
            xy.append(vv[0])
        paths.append(np.vstack(xy))
    contours.append(paths)
contours = contours[0][0]

axs.fill_between(contours[:,0],contours[:,1],np.max(contours[:,1])*np.ones_like(contours[:,0]),color=(1,0,0),alpha=0.3)

fig.tight_layout()
fig.show()
fig.savefig("1_example.png",dpi=300)

print("stop")