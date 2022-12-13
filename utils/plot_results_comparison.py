import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tikzplotlib

from matplotlib import rcParams
rcParams['text.usetex'] = True

#meta_test data
beta = np.array([0.5,0.2,0.1,0.05])

grid_J = np.array([0.0059,0.0648,0.3256,0.8301])
grid_MR = np.array([0.0125,0.0225,0.0275,0.0800])
grid_prec = np.array([0.9867,0.9273,0.8941,0.6961])
grid_gamma = np.array([2024,2201,2135,2488])/20000
grid_failures = np.array([1997,2041,1909,1732])

fc_J = np.array([0.0053,0.0316,0.1720,0.6804])
fc_MR = np.array([0.0025,0.0100,0.0325,0.0650])
fc_prec = np.array([0.9857,0.9648,0.9133,0.7287])
fc_gamma = np.array([2094,2046,2098,2440])/20000
fc_failures = np.array([2064,1974,1916,1778])

nfc_J = np.array([0.0058,0.0210,0.0661,1.2059])
nfc_MR = np.array([0.0050,0.0075,0.0225,0.0900])
nfc_prec = np.array([0.9884,0.9675,0.9390,0.6703])
nfc_gamma = np.array([2074,2061,2130,2551])/20000
nfc_failures = np.array([2050,1994,2000,1710])

# c = [(10.09/255,155.49/255,10.09/255),(183.03/255,25.40/255,183.03/255),(229.02/255,134.18/255,39.34/255),(64.88/255,125.39/255,185.90/255)]
c = [(182.91/255,53.29/255,85.70/255),(65.28/255,17.26/255,209.35/255),(149.47/255,184.89/255,43.21/255)]

legend_elements = [Line2D([0], [0], color=c[0],linestyle="solid", label=r'$\mathrm{Grid~Search}$'),
                   Line2D([0], [0], color=c[1],linestyle="solid", label=r'$\mathrm{BO~Failure~Trajectories}$'),
                   Line2D([0], [0], color=c[2],linestyle="solid", label=r'$\mathrm{BO~All~Trajectories}$')]

#Trajectory similarity
fig,ax = plt.subplots(1,1,figsize=(5,2.5))

ax.semilogy(beta,grid_J,color=c[0],marker="s")
ax.semilogy(beta,fc_J,color=c[1],marker="s")
ax.semilogy(beta,nfc_J,color=c[2],marker="s")
ax.grid(which="both")

ax.set_xlabel(r"$\beta$",labelpad=-1)
ax.set_ylabel(r"$J_{test}$",labelpad=3)

ax.legend(handles=legend_elements,loc="upper right",fancybox=False,edgecolor="none")#.get_frame().set_boxstyle('Round', pad=0.2, rounding_size=0.3)

fig.tight_layout()
fig.savefig("./figs/comparison_J.pdf",dpi=600,bbox_inches="tight", pad_inches=0.02)


#Miss Rate
fig,ax = plt.subplots(1,1,figsize=(5,2.5))

ax.plot(beta,grid_MR,color=c[0],marker="s")
ax.plot(beta,fc_MR,color=c[1],marker="s")
ax.plot(beta,nfc_MR,color=c[2],marker="s")
ax.grid(which="both")

ax.set_xlabel(r"$\beta$",labelpad=-1)
ax.set_ylabel(r"$MR_{test}$",labelpad=3)

ax.legend(handles=legend_elements,loc="upper right",fancybox=False,edgecolor="none")#.get_frame().set_boxstyle('Round', pad=0.2, rounding_size=0.3)
ax.set_ylim([0.0,0.1])

fig.tight_layout()
fig.savefig("./figs/comparison_MR.pdf",dpi=600,bbox_inches="tight", pad_inches=0.02)

#Precision
fig,ax = plt.subplots(1,1,figsize=(5,2.5))

ax.plot(beta,grid_prec,color=c[0],marker="s")
ax.plot(beta,fc_prec,color=c[1],marker="s")
ax.plot(beta,nfc_prec,color=c[2],marker="s")
ax.grid(which="both")

ax.set_xlabel(r"$\beta$",labelpad=-1)
ax.set_ylabel(r"$\mathrm{Precision}_{verf.}$",labelpad=3)

ax.legend(handles=legend_elements,loc="lower right",fancybox=False,edgecolor="none")#.get_frame().set_boxstyle('Round', pad=0.2, rounding_size=0.3)
# ax.set_yticks(np.arange(0.65,1.0,0.05))
ax.set_ylim([0.60,1.0])

fig.tight_layout()
fig.savefig("./figs/comparison_prec.pdf",dpi=600,bbox_inches="tight", pad_inches=0.02)


#Found Failures
fig,ax = plt.subplots(1,1,figsize=(5,2.5))

#cost per failure: 20000*(b*c_hf+g*c_h)/failures_found
#                  20000*c_hf/failures_found

ax.plot(beta,(1-grid_MR)/(beta+grid_gamma),color=c[0],marker="s")
ax.plot(beta,(1-fc_MR)/(beta+fc_gamma),color=c[1],marker="s")
ax.plot(beta,(1-nfc_MR)/(beta+nfc_gamma),color=c[2],marker="s")
ax.grid(which="both")

# ax.plot(beta,1/(beta+grid_gamma),color=c[0],marker="s")
# ax.plot(beta,1/(beta+fc_gamma),color=c[1],marker="s")
# ax.plot(beta,1/(beta+nfc_gamma),color=c[2],marker="s")
# ax.grid(which="both")

ax.set_xlabel(r"$\beta$",labelpad=-1)
ax.set_ylabel(r"$\mathrm{Relative~Found~Failures}$",labelpad=3)

ax.legend(handles=legend_elements,loc="upper right",fancybox=False,edgecolor="none")#.get_frame().set_boxstyle('Round', pad=0.2, rounding_size=0.3)
# ax.set_yticks(np.arange(0.65,1.0,0.05))
ax.set_ylim([0,6])

fig.tight_layout()
fig.savefig("./figs/comparison_found_failures.pdf",dpi=600,bbox_inches="tight", pad_inches=0.02)


print("stop")