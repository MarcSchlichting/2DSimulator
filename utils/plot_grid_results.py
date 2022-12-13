import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tikzplotlib

from matplotlib import rcParams
rcParams['text.usetex'] = True


LEVEL_1 = 1750
LEVEL_2 = 700
LEVEL_3 = 350
LEVEL_4 = 175

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

c = [(10.09/255,155.49/255,10.09/255),(183.03/255,25.40/255,183.03/255),(229.02/255,134.18/255,39.34/255),(64.88/255,125.39/255,185.90/255)]

legend_elements = [Line2D([0], [0], color=c[0],linestyle="solid", label=r'$\beta=0.50$'),
                   Line2D([0], [0], color=c[1],linestyle="solid", label=r'$\beta=0.20$'),
                   Line2D([0], [0], color=c[2],linestyle="solid", label=r'$\beta=0.10$'),
                   Line2D([0], [0], color=c[3],linestyle="solid", label=r'$\beta=0.05$')]


#########################RK4##############################
fig,ax = plt.subplots(1,1,figsize=(5,2.5))

pcm = ax.imshow(np.mean(rk4_data,axis=-1),vmin=0,vmax=500,extent=[0.1-(9.9/38),10+(9.9/38),0.1-(1.4/38),1.5+(1.4/38)],origin="lower",aspect="auto",cmap="Greys_r")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_1],colors=[(47.35/255,201.48/255,62.77/255)],linestyles="solid",label=r"$\beta=0.5$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_2],colors=[(129.79/255,12.62/255,118.07/255)],linestyles="solid",label=r"$\beta=0.2$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_3],colors=[(197.83/255,134.48/255,39.48/255)],linestyles="solid",label=r"$\beta=0.1$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_4],colors=[(0.0/255,63.75/255,127.5/255)],linestyles="solid",label=r"$\beta=0.05$")

ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_1],colors=[c[0]],linestyles="solid",label=r"$\beta=0.5$")
ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_2],colors=[c[1]],linestyles="solid",label=r"$\beta=0.2$")
ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_3],colors=[c[2]],linestyles="solid",label=r"$\beta=0.1$")
ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_4],colors=[c[3]],linestyles="solid",label=r"$\beta=0.05$")
cb = fig.colorbar(pcm, ax=ax)
cb.set_label(r'$J$',rotation=0,loc="center")

ax.set_xlim([0.1,10.0])
ax.set_ylim([0.1,1.5])

ax.set_xlabel(r"$\sigma_{sensor}$",labelpad=-3)
ax.set_ylabel(r"$dt$",labelpad=1)
ax.legend(handles=legend_elements,loc="upper right",fancybox=False,edgecolor="none")#.get_frame().set_boxstyle('Round', pad=0.2, rounding_size=0.3)
# ax.legend().get_frame().set_boxstyle('Round', pad=0.2, rounding_size=0.3)
fig.tight_layout()
fig.savefig("./figs/grid_rk4.pdf",dpi=600,bbox_inches="tight", pad_inches=0.02)


#########################RK2##############################
fig,ax = plt.subplots(1,1,figsize=(5,2.5))

pcm = ax.imshow(np.mean(rk2_data,axis=-1),vmin=0,vmax=500,extent=[0.1-(9.9/38),10+(9.9/38),0.1-(1.4/38),1.5+(1.4/38)],origin="lower",aspect="auto",cmap="Greys_r")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_1],colors=[(47.35/255,201.48/255,62.77/255)],linestyles="solid",label=r"$\beta=0.5$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_2],colors=[(129.79/255,12.62/255,118.07/255)],linestyles="solid",label=r"$\beta=0.2$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_3],colors=[(197.83/255,134.48/255,39.48/255)],linestyles="solid",label=r"$\beta=0.1$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_4],colors=[(0.0/255,63.75/255,127.5/255)],linestyles="solid",label=r"$\beta=0.05$")

ax.contour(S_STD_eval,DT_eval,cost_rk2,levels=[LEVEL_1],colors=[c[0]],linestyles="solid",label=r"$\beta=0.5$")
ax.contour(S_STD_eval,DT_eval,cost_rk2,levels=[LEVEL_2],colors=[c[1]],linestyles="solid",label=r"$\beta=0.2$")
ax.contour(S_STD_eval,DT_eval,cost_rk2,levels=[LEVEL_3],colors=[c[2]],linestyles="solid",label=r"$\beta=0.1$")
ax.contour(S_STD_eval,DT_eval,cost_rk2,levels=[LEVEL_4],colors=[c[3]],linestyles="solid",label=r"$\beta=0.05$")

ax.scatter([0.1],[0.1737],marker="s",s=30,color=c[0])
ax.scatter([1.1421],[0.2474],marker="s",s=30,color=c[2])
ax.set_xlim([0.1,10.0])
ax.set_ylim([0.1,1.5])

cb = fig.colorbar(pcm, ax=ax)
cb.set_label(r'$J$',rotation=0,loc="center")


ax.set_xlabel(r"$\sigma_{sensor}$",labelpad=-3)
ax.set_ylabel(r"$dt$",labelpad=1)
ax.legend(handles=legend_elements,loc="upper right",fancybox=False,edgecolor="none")#.get_frame().set_boxstyle('Round', pad=0.2, rounding_size=0.3)
fig.tight_layout()
fig.savefig("./figs/grid_rk2.pdf",dpi=600,bbox_inches="tight", pad_inches=0.02)

#########################RK1##############################
fig,ax = plt.subplots(1,1,figsize=(5,2.5))

pcm = ax.imshow(np.mean(rk1_data,axis=-1),vmin=0,vmax=500,extent=[0.1-(9.9/38),10+(9.9/38),0.1-(1.4/38),1.5+(1.4/38)],origin="lower",aspect="auto",cmap="Greys_r")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_1],colors=[(47.35/255,201.48/255,62.77/255)],linestyles="solid",label=r"$\beta=0.5$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_2],colors=[(129.79/255,12.62/255,118.07/255)],linestyles="solid",label=r"$\beta=0.2$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_3],colors=[(197.83/255,134.48/255,39.48/255)],linestyles="solid",label=r"$\beta=0.1$")
# ax.contour(S_STD_eval,DT_eval,cost_rk4,levels=[LEVEL_4],colors=[(0.0/255,63.75/255,127.5/255)],linestyles="solid",label=r"$\beta=0.05$")

ax.contour(S_STD_eval,DT_eval,cost_rk1,levels=[LEVEL_1],colors=[c[0]],linestyles="solid",label=r"$\beta=0.5$")
ax.contour(S_STD_eval,DT_eval,cost_rk1,levels=[LEVEL_2],colors=[c[1]],linestyles="solid",label=r"$\beta=0.2$")
ax.contour(S_STD_eval,DT_eval,cost_rk1,levels=[LEVEL_3],colors=[c[2]],linestyles="solid",label=r"$\beta=0.1$")
ax.contour(S_STD_eval,DT_eval,cost_rk1,levels=[LEVEL_4],colors=[c[3]],linestyles="solid",label=r"$\beta=0.05$")

ax.scatter([0.6211],[0.1],marker="s",s=30,color=c[1])
ax.scatter([0.6211],[0.3947],marker="s",s=30,color=c[3])
ax.set_xlim([0.1,10.0])
ax.set_ylim([0.1,1.5])

cb = fig.colorbar(pcm, ax=ax)
cb.set_label(r'$J$',rotation=0,loc="center")

ax.set_xlabel(r"$\sigma_{sensor}$",labelpad=-3)
ax.set_ylabel(r"$dt$",labelpad=1)
ax.legend(handles=legend_elements,loc="upper right",fancybox=False,edgecolor="none")#.get_frame().set_boxstyle('Round', pad=0.2, rounding_size=0.3)
fig.tight_layout()
fig.savefig("./figs/grid_rk1.pdf",dpi=600,bbox_inches="tight", pad_inches=0.02)


# tikzplotlib.save("test.tex")
print("stop")
