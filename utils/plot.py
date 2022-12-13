import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def plot_two_scenarios(results1, results2, experiment_label_1, experiment_label_2, title, fname):
    run1_x = results1["x_pos"][1]
    run1_y = results1["y_pos"][1]
    run1_t = results1["t"]
    run1_collision = results1["collision"]
    run2_x = results2["x_pos"][1]
    run2_y = results2["y_pos"][1]
    run2_collision = results2["collision"]
    run2_t = results2["t"]

    run1_collision_first_idx = run1_collision.shape[0] if np.all(np.logical_not(run1_collision)) else np.nanmin(np.where(run1_collision>0,np.arange(run1_collision.shape[0]),np.nan)).astype(np.int32)
    run2_collision_first_idx = run2_collision.shape[0] if np.all(np.logical_not(run2_collision)) else np.nanmin(np.where(run2_collision>0,np.arange(run2_collision.shape[0]),np.nan)).astype(np.int32)

    plt.rcParams["text.usetex"] = True
    # matplotlib.use('pgf')
    plt.plot(run1_t[:run1_collision_first_idx+1],run1_x[:run1_collision_first_idx+1],"--",label=experiment_label_1,color=(0,0,0))
    plt.plot(run1_t[run1_collision_first_idx:],run1_x[run1_collision_first_idx:],"--",color=(1,0,0))
    plt.plot(run2_t[:run2_collision_first_idx+1],run2_x[:run2_collision_first_idx+1],":",label=experiment_label_2,color=(0,0,0))
    plt.plot(run2_t[run2_collision_first_idx:],run2_x[run2_collision_first_idx:],":",color=(1,0,0))
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$x$ [m]")
    plt.title(title)
    plt.legend()
    plt.grid(which="both")
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname,dpi=600)
    print("stop")
    