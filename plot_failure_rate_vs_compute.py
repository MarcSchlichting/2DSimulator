import numpy as np
import matplotlib.pyplot as plt

c = np.linspace(0,1,1001)
mr = np.linspace(0,0.4,1001)
C,MR = np.meshgrid(c,mr)

Z = (1-MR)/C

fig,ax = plt.subplots(1,1,figsize=(8,3))
ax.contourf(C,MR,Z,vmin=0,vmax=10,levels=[0.0,1.0,2.0,5.0,10.0,1000.0],alpha=0.7)
c = ax.contour(C,MR,Z,levels=[0.0,1.0,2.0,5.0,10.0,1000.0])
ax.scatter([0.5,0.2,0.1,0.05],[0.0025,0.0075,0.0225,0.0650],marker="s",color="red",label="Optimization Results")
ax.clabel(c,inline=True,fontsize=10,manual=True)
for x,y,s in zip([0.5,0.2,0.1,0.05],[0.0025,0.0075,0.0225,0.0650],[(1-0.0025)/0.5,(1-0.0075)/0.2,(1-0.0225)/0.1,(1-0.0650)/0.05]):
    ax.annotate(f"{s:.1f}",np.array([x,y])*1.05,color="red")

# ax.contour(C,MR,Z,levels=[1.0],linewidth=3,color=(1,0,0))
# ax.set_colorbar()
ax.set_xlabel("Relative Compute Cost")
ax.set_ylabel("Miss Rate")
ax.legend(loc="upper right")
fig.tight_layout()
fig.show()
fig.savefig("found_failures.png",dpi=600)
print("stop")