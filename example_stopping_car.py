from asyncio import run
from multiprocessing.context import SpawnProcess
import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
from idm import idm_driver
import torch
import plot
import time
from ax.service.ax_client import AxClient

def run_scenario(dt,integration_method,sensor_std):

    dt = dt # time steps in terms of seconds. In other words, 1/dt is the FPS.
    w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

    # Let's add some sidewalks and RectangleBuildings.
    # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
    # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
    # For both of these objects, we give the center point and the size.
    w.add(Painting(Point(60, 106.5), Point(120, 27), 'gray80')) # We build a sidewalk.
    w.add(RectangleBuilding(Point(60, 107.5), Point(120, 25))) # The RectangleBuilding is then on top of the sidewalk, with some margin.

    w.add(Painting(Point(60, 41), Point(120, 82), 'gray80')) # We build a sidewalk.
    w.add(RectangleBuilding(Point(60, 40), Point(120, 80))) # The RectangleBuilding is then on top of the sidewalk, with some margin.



    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    c1= Car(Point(80,90), np.pi, 'blue')
    c1.velocity = Point(2.8,0.0) # We can also specify an initial velocity just like this.
    c1.integration_method = integration_method
    c1.max_break_acceleration = -10
    w.add(c1)

    c2 = Car(Point(110,90), np.pi)
    c2.integration_method = integration_method
    c2.velocity = Point(20.0,0)
    c2.max_speed = 10
    c2.max_break_acceleration = -1.5
    c2.position_sensor_std = sensor_std
    c2.velocity = Point(3.0,0.)

    w.add(c2)


    # w.render() # This visualizes the world we just constructed.

    #datalogs
    x_pos_list = []
    y_pos_list = []
    speed_list = []
    acc_list = []
    collision_list = []
    t_list = []

    # Let's implement some simple scenario with all agents
    c1.set_control(0, 0.5)
    c2.set_control(0, 2.0)
    for k in range(int(25/dt)):
        if k==0:
            x_pos_list.append([c1.x,c2.x])
            y_pos_list.append([c1.y,c2.y])
            speed_list.append([c1.speed,c2.speed])
            acc_list.append([c1.acceleration,c2.acceleration])
            collision_list.append(False)
            t_list.append(k*dt)

        c2.set_control(*idm_driver(w,c2,c1))
        # All movable objects will keep their control the same as long as we don't change it.
        if k*dt >= 13.0:    #after 13s of simulation time 
            c1.set_control(0,-5.0)
        w.tick() # This ticks the world for one time step (dt second)
        # w.render()
        # time.sleep(dt/16) # Let's watch it 4x

        collision = w.collision_exists()
        # if collision: # Or we can check if there is any collision at all.
        #     print('Collision exists somewhere...')

        #log all the data
        x_pos_list.append([c1.x,c2.x])
        y_pos_list.append([c1.y,c2.y])
        speed_list.append([c1.speed,c2.speed])
        acc_list.append([c1.acceleration,c2.acceleration])
        collision_list.append(collision)
        t_list.append((k+1)*dt)

    results_dict = {"x_pos":np.array(x_pos_list).T,
                    "y_pos":np.array(y_pos_list).T,
                    "speed":np.array(speed_list).T,
                    "acc":np.array(acc_list).T,
                    "collision":np.array(collision_list),
                    "t":np.array(t_list)}


    w.close()

    return results_dict

def compare_trajectories(results1, results2):
    """Compare the trajectories of two runs and compute the MSE between the two trajectories (if two different sampling frequency, the higher frequency is downsampled)
    """
    run1_x = results1["x_pos"][1]
    run1_y = results1["y_pos"][1]
    run1_collision = results1["collision"].astype(np.int32)
    run1_t = results1["t"]
    run2_x = results2["x_pos"][1]
    run2_y = results2["y_pos"][1]
    run2_collision = results2["collision"].astype(np.int32)
    run2_t = results2["t"]

    #downsample
    if run1_t.shape[0] != run2_t.shape[0]:
        if run1_t.shape[0] > run2_t.shape[0]:
            t_common = run2_t
            run1_x = np.interp(t_common,run1_t,run1_x)
            run1_y = np.interp(t_common,run1_t,run1_y)
            run1_collision = np.interp(t_common,run1_t,run1_collision)
        else:
            t_common = run1_t
            run2_x = np.interp(t_common,run2_t,run2_x)
            run2_y = np.interp(t_common,run2_t,run2_y)
            run2_collision = np.interp(t_common,run2_t,run2_collision)
         
    else:
        t_common = run1_t

    #compute MSE
    run1_pos = np.stack([run1_x,run1_y])
    run2_pos = np.stack([run2_x,run2_y])
    #MSE for trajectories
    MSE = (np.linalg.norm(run1_pos-run2_pos,axis=0)**2).mean()
    #BCE for collision
    BCE = torch.nn.functional.binary_cross_entropy(torch.Tensor(run1_collision),torch.Tensor(run2_collision))
    
    return MSE,BCE
class AXWrapper():
    def __init__(self) -> None:
        self.ref_results = run_scenario(0.1,"RK4",0)

    def ax_wrapper(self, x_dict):
        integration_method_cost = {"Forward_Euler":1, "RK2":2, "RK4":4}
        COST= 1/x_dict["dt"] + integration_method_cost[x_dict["integration_method"]] + 1/x_dict["sensor_std"]
        res = run_scenario(**x_dict)
        MSE,BCE = compare_trajectories(self.ref_results, res)
        return {"cost": (MSE, 0.0), "compute_cost":(COST, 0.0) }

def ax_search():
    ax_client = AxClient()
    ax_client.create_experiment(
    name="carlo_optimization",
    parameters=[
        {
            "name": "dt",
            "type": "range",
            "bounds": [0.1, 2.5],
            "value_type": "float", 
            "log_scale": False, 
        },
        {
            "name": "integration_method",
            "type": "choice",
            "values": ["Forward_Euler", "RK2", "RK4"],
            "value_type": "str",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  
        },
        {
            "name": "sensor_std",
            "type": "range",
            "bounds": [1e-05, 10.0],
            "value_type": "float", 
            "log_scale": True,  
        },
    ],
    objective_name="cost",
    minimize=True,  # Optional, defaults to False.
    outcome_constraints = ["compute_cost <= 5"]
    )

    axw = AXWrapper()
    for i in range(25):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=axw.ax_wrapper(parameters))
    
    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters)
    print(values)


if __name__=="__main__":
    import matplotlib.pyplot as plt
    ax_search()

    # #compare against standard case dt=0.1
    # MSEs = []
    # BCEs =[]
    # dts = np.arange(0.1,2.1,0.1)
    # for dt in dts:
    #     res1 = run_scenario(0.1,"Forward_Euler",0)
    #     res2 = run_scenario(dt,"Forward_Euler",0)
    #     MSE,BCE = compare_trajectories(res1,res2)
    #     MSEs.append(MSE)
    #     BCEs.append(BCE)

    # plt.rcParams["text.usetex"] = True
    # fig,axs = plt.subplots(2,1,figsize=(7,6))
    # axs[0].plot(dts,MSEs,color=(0,0,0))
    # axs[0].set_xlabel(r"Step Width $dt$ [s]")
    # axs[0].set_ylabel(r"MSE Position")
    # axs[0].set_xlim(dts.min(),dts.max())
    # axs[0].grid(which="both")
    # axs[1].plot(dts,BCEs,color=(0,0,0))
    # axs[1].set_xlabel(r"Step Width $dt$ [s]")
    # axs[1].set_ylabel(r"BCE Collision")
    # axs[1].set_xlim(dts.min(),dts.max())
    # axs[1].grid(which="both")
    # fig.suptitle("Comparison Step Width For Euler's Method And No Noise")
    # fig.tight_layout()
    # fig.savefig("./figs/comparison_dt_euler.png",dpi=600)
    # plt.clf()
    # print("stop")

    # res1 = run_scenario(0.1,"RK4",0)
    # res2 = run_scenario(2.5,"RK4",0)
    # plot.plot_two_scenarios(res1,res2, r"$dt=0.1s$", r"$dt=2.5s$", "Trajectory Comparison RK4", "./figs/comparison_dt_rk4_trajectories.png")

    # #compare RK4 vs Euler Forward for different dt
    # MSEs = []
    # BCEs = []
    # dts = np.arange(0.1,2.1,0.1)
    # for dt in dts:
    #     res1 = run_scenario(dt,"RK4",0)
    #     res2 = run_scenario(dt,"Forward_Euler",0)
    #     MSE,BCE = compare_trajectories(res1,res2)
    #     MSEs.append(MSE)
    #     BCEs.append(BCE)
    
    # fig,axs = plt.subplots(2,1,figsize=(10,10))
    # axs[0].plot(dts,MSEs)
    # axs[1].plot(dts,BCEs)
    # fig.show()
    # print("stop")

    # res1 = run_scenario(1.0,"RK1",0)
    # res2 = run_scenario(1.0,"Forward_Euler",0)
    # plot.plot_two_scenarios(res1,res2, r"RK1", r"Euler", "Trajectory Comparison RK1  vs. RK4", "./figs/comparison_rk1_rk4_trajectories.png")


    # # #compare different sensor noise for RK4 method and dt=0.1
    # MSEs = []
    # BCEs = []
    # stds = np.logspace(-1,1,20,base=10)
    # for std in stds:
    #     res1 = run_scenario(0.1,"RK4",0)
    #     res2 = run_scenario(0.1,"RK4",std)
    #     MSE,BCE = compare_trajectories(res1,res2)
    #     MSEs.append(MSE)
    #     BCEs.append(BCE)
    
    # fig,axs = plt.subplots(2,1,figsize=(10,10))
    # axs[0].semilogx(stds,MSEs)
    # axs[1].semilogx(stds,BCEs)
    # fig.show()
    # print("stop")
    
    # results1 = run_scenario(0.1,"Heun",0)
    # results2 = run_scenario(0.1,"Forward_Euler",0)
    # plot.plot_two_scenarios(results1,results2)

    # #compare Euler and RK4
    # results1 = run_scenario(0.1,"Forward_Euler",0)
    # results2 = run_scenario(0.1,"Forward_Euler",2.0)
    # plot.plot_two_scenarios(results1,results2, "std=0.0", "std=1.0")

    # #time
    # st1 = time.process_time()
    # for _ in range(10):
    #     _ = run_scenario(0.1,"Forward_Euler",0)
    # et1 = time.process_time()
    # diff1 = et1-st1
    # st2 = time.process_time()
    # for _ in range(10):
    #     _ = run_scenario(0.1,"RK2",0)
    # et2 = time.process_time()
    # diff2 = et2-st2
    # print(diff1,diff2)


    print("stop")




    

