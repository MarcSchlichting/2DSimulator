import numpy as np
import matplotlib.pyplot as plt

rk4_data = np.load("RK4_all.npy")
rk2_data = np.load("RK2_all.npy")
fe_data = np.load("Forward_Euler_all.npy")

fig,axs = plt.subplots(1,3,figsize=(15,5))
axs[0].matshow(np.mean(rk4_data,axis=-1),vmin=0,vmax=1000)
axs[0].set_title("RK4")
axs[1].matshow(np.mean(rk2_data,axis=-1),vmin=0,vmax=1000)
axs[1].set_title("RK2")
axs[2].matshow(np.mean(fe_data,axis=-1),vmin=0,vmax=1000)
axs[2].set_title("Forward Euler")

print("stop")