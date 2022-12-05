import numpy as np
import matplotlib.pyplot as plt

c = np.linspace(0,1,1001)
mr = np.linspace(0,0.5,1001)
C,MR = np.meshgrid(c,mr)

Z = (1-MR)/C

fig,ax = plt.subplots(1,1,figsize=(5,2.5))
ax.contourf(C,MR,Z,vmin=0,vmax=10,levels=[0.0,1.0,2.0,5.0,10.0,1000.0],alpha=0.7)

c = ax.contour(C,MR,Z,levels=[0.0,1.0,2.0,5.0,10.0,1000.0])
ax.clabel(c,inline=True,fontsize=10,manual=True)
# ax.contour(C,MR,Z,levels=[1.0],linewidth=3,color=(1,0,0))
# ax.set_colorbar()
ax.set_xlabel("Relative Compute Cost")
ax.set_ylabel("Miss Rate")
fig.tight_layout()
fig.savefig("found_failures.png",dpi=600)
print("stop")