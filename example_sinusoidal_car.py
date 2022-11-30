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
import math
import time
# from mpire import WorkerPool

class SinusoidalCarScenario(object):
    def __init__(self) -> None:
        #default configuration for inner loop
        self.name = "SinusoidalCarScenario"
        self.scenario_configuration = {"ego_max_speed":10,
                                    "ego_max_break_acceleration":-0.8,
                                    "ego_initial_speed":3.0,
                                    "lead_car_initial_speed":3.0,
                                    "lead_car_max_speed":10.0,
                                    "lead_car_acc":2.0,
                                    "lead_car_break_acc":-8.0,
                                    "lead_car_osc_period":5,
                                    "idm_T":1.0,
                                    "idm_a":1.5,
                                    "idm_b":1.5,
                                    "idm_s0":1.1}

        #define the type for each of the variables
        self.scenario_configuration_domain = {"ego_max_speed":{"type":"range", "values":[5,20]},
                                    "ego_max_break_acceleration":{"type":"range", "values":[-4,-0.5]},
                                    "ego_initial_speed":{"type":"range", "values":[3,10]},
                                    "lead_car_initial_speed":{"type":"range","values":[1.0,5.0]},
                                    "lead_car_max_speed":{"type":"range","values":[5.0,15.0]},
                                    "lead_car_acc":{"type":"range","values":[0.5,10.0]},
                                    "lead_car_break_acc":{"type":"range","values":[-10.0,-4.0]},
                                    "lead_car_osc_period":{"type":"range","values":[1.0,7.0]},
                                    "idm_T":{"type":"range", "values":[0.5,3.0]},
                                    "idm_a":{"type":"range", "values":[0.5,2.0]},
                                    "idm_b":{"type":"range", "values":[0.5,2.0]},
                                    "idm_s0":{"type":"range", "values":[0.0,5.0]}}
    
    def sample_scenario_configuration(self):
        """Method that randomly samples from all the possible scenario configurations.

        Raises:
            NotImplementedError: If type of the parameters is not "range", sampling from those scenarios has not yet been implemented.

        Returns:
            Dict: Dictionary that contains the sampled values for the scenario.
        """

        sampled_scenario_configuration = {}
        for k in self.scenario_configuration.keys():
            if self.scenario_configuration_domain[k]["type"]=="range":
                sampled_value = (torch.rand(1)*(self.scenario_configuration_domain[k]["values"][1]-self.scenario_configuration_domain[k]["values"][0]) + self.scenario_configuration_domain[k]["values"][0])
                sampled_scenario_configuration[k] = sampled_value
            else:
                raise NotImplementedError
        
        return sampled_scenario_configuration

    def run(self,scenario_configuration, simulation_configuration,render=False):
        """Function that runs the scenario.

        Args:
            scenario_configuration (dict): Dictionary that contains all the values for the scenario configuration specified in the __init__ function. Values should be within the ranges specified in self.scenario_configuration_domain.
            simulation_configurations (dict): Dictionary that contains all the parameters for the simulation (i.e., dt, integration_method, sensor_std)
        Returns:
            dict: Dictionary containing the trajectory of x-position, y-position, speed, acceleration, the simulation time, and if a collision is occuring at a given time point.
        """

        dt = simulation_configuration["dt"] # time steps in terms of seconds. In other words, 1/dt is the FPS.
        w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

        # Let's add some sidewalks and RectangleBuildings.
        # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
        # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
        # For both of these objects, we give the center point and the size.
        w.add(Painting(Point(60, 106.5), Point(120, 27), 'gray80')) # We build a sidewalk.
        w.add(RectangleBuilding(Point(60, 107.5), Point(120, 25))) # The RectangleBuilding is then on top of the sidewalk, with some margin.

        w.add(Painting(Point(60, 41), Point(120, 82), 'gray80')) # We build a sidewalk.
        w.add(RectangleBuilding(Point(60, 40), Point(120, 80))) # The RectangleBuilding is then on top of the sidewalk, with some margin.



        # The lead car
        c1= Car(Point(80,90), np.pi, 'blue')
        c1.velocity = Point(scenario_configuration["lead_car_initial_speed"],0.0) 
        c1.integration_method = simulation_configuration["integration_method"]
        c1.max_speed = scenario_configuration["lead_car_max_speed"] 
        c1.max_break_acceleration = scenario_configuration["lead_car_break_acc"]
        w.add(c1)

        c2 = Car(Point(110,90), np.pi)
        c2.integration_method = simulation_configuration["integration_method"]
        c2.max_speed = scenario_configuration["ego_max_speed"] #10
        c2.max_break_acceleration = scenario_configuration["ego_max_break_acceleration"] #-1.5
        c2.position_sensor_std = simulation_configuration["sensor_std"]
        c2.velocity = Point(scenario_configuration["ego_initial_speed"],0.0) #Point(3.0,0.0)
        w.add(c2)

        c_inf = Car(Point(-10000,90),np.pi)
        c_inf.integration_method = simulation_configuration["integration_method"]
        w.add(c_inf)

        #datalogs
        x_pos_list = []
        y_pos_list = []
        speed_list = []
        acc_list = []
        collision_list = []
        t_list = []

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

            c2.set_control(*idm_driver(w,c2,c1,scenario_configuration["idm_s0"],scenario_configuration["idm_T"],scenario_configuration["idm_a"],scenario_configuration["idm_b"]))
            
            if math.sin(2*math.pi/scenario_configuration["lead_car_osc_period"]*k*simulation_configuration["dt"]) > 0:
                c1_control = (0,scenario_configuration["lead_car_acc"])
            else:
                c1_control = (0,scenario_configuration["lead_car_break_acc"])
            

            c1.set_control(*c1_control)
            
            w.tick() 
            if render:
                w.render()
                time.sleep(dt/16) # Let's watch it 4x

            collision = w.collision_exists()

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
    
    def inner_loop_one_step(self,simulation_configuration):
        sampled_scenario_config = self.sample_scenario_configuration()
            
        result = self.run(sampled_scenario_config,simulation_configuration)
        
        failure = np.any(result["collision"])
        print(f"Collision found: {failure}")
        if failure:
            return result,sampled_scenario_config,None
        else:
            return None,None,sampled_scenario_config

    def inner_loop_mc(self,simulation_configuration,num_trials,num_processes=1):
        """Random trials for values sampled from the inner loop.

        Args:
            simulation_configuration (Dict): Dictionary containing all the simulation parameters (i.e., dt, integration_method, and sensor_std)
            num_trials (int): Number of trials

        Returns:
            Tuple: Tuple of lists of dictionaries that contain the trajectory of failures, the scenario configurations that led to a failure, and the scenario configurations that didn't lead to a failure
        """
        self.current_simulation_configuration = simulation_configuration
        
        failures = []
        failure_configs = []
        non_failure_configs = []

        with WorkerPool(n_jobs=num_processes) as pool:

            for result in pool.imap(self.inner_loop_one_step, num_trials * [(simulation_configuration,)],progress_bar=True):
            # result = self.inner_loop_one_step()

                if result[0] is not None:
                    failures.append(result[0])
                if result[1] is not None:
                    failure_configs.append(result[1])
                if result[2] is not None:
                    non_failure_configs.append(result[2])
        

        for i in range(num_trials):
            print(i)
            result = self.inner_loop_one_step(simulation_configuration)

            if result[0] is not None:
                failures.append(result[0])
            if result[1] is not None:
                failure_configs.append(result[1])
            if result[2] is not None:
                non_failure_configs.append(result[2])
            
        
        return failures, failure_configs, non_failure_configs
    
    def plot_inner_loop_results(self, failure_configs, non_failure_configs):

        num_scenario_parameters = len(self.scenario_configuration)
        scenario_parameters = list(self.scenario_configuration.keys())
        num_cols = min(5,num_scenario_parameters)
        num_rows = math.ceil(num_scenario_parameters/num_cols)

        N_bins = 10
        plt.rcParams['text.usetex'] = True
        fig,axs = plt.subplots(num_rows,num_cols,figsize=(5*num_cols,5*num_rows))
        axs = axs.flatten()

        for i in range(num_scenario_parameters):
            #get data
            failures = torch.Tensor([f[scenario_parameters[i]] for f in failure_configs])
            non_failures = torch.Tensor([f[scenario_parameters[i]]for f in non_failure_configs])

            #plot the data
            axs[i].hist(non_failures,bins=N_bins,color=(0,0.8,0.2),alpha=0.5)
            axs[i].hist(failures,bins=N_bins,color=(1,0,0),alpha=0.5)
            axs[i].set_title(scenario_parameters[i])
       
        fig.tight_layout()
        fig.savefig(f"MC_inner_loop_hist_sinusoidal_{int(time.time())}.png",dpi=600)
        # fig.show()
        print("stop")

























# def run_scenario(dt,integration_method,sensor_std,idm_configuration={"ego_max_speed":10,"ego_max_break_acceleration":-1.5,"ego_initial_speed":3,"idm_T":1.0,"idm_a":1.5,"idm_b":1.5,"idm_s0":1.1}):

#     dt = dt # time steps in terms of seconds. In other words, 1/dt is the FPS.
#     w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

#     # Let's add some sidewalks and RectangleBuildings.
#     # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
#     # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
#     # For both of these objects, we give the center point and the size.
#     w.add(Painting(Point(60, 106.5), Point(120, 27), 'gray80')) # We build a sidewalk.
#     w.add(RectangleBuilding(Point(60, 107.5), Point(120, 25))) # The RectangleBuilding is then on top of the sidewalk, with some margin.

#     w.add(Painting(Point(60, 41), Point(120, 82), 'gray80')) # We build a sidewalk.
#     w.add(RectangleBuilding(Point(60, 40), Point(120, 80))) # The RectangleBuilding is then on top of the sidewalk, with some margin.



#     # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
#     c1= Car(Point(80,90), np.pi, 'blue')
#     c1.velocity = Point(2.8,0.0) # We can also specify an initial velocity just like this.
#     c1.integration_method = integration_method
#     c1.max_break_acceleration = -10
#     w.add(c1)

#     c2 = Car(Point(110,90), np.pi)
#     c2.integration_method = integration_method
#     # c2.velocity = Point(20.0,0)
#     c2.max_speed =idm_configuration["ego_max_speed"] #10
#     c2.max_break_acceleration = idm_configuration["ego_max_break_acceleration"] #-1.5
#     c2.position_sensor_std = sensor_std
#     c2.velocity = Point(idm_configuration["ego_initial_speed"],0.0) #Point(3.0,0.)

#     w.add(c2)


#     # w.render() # This visualizes the world we just constructed.

#     #datalogs
#     x_pos_list = []
#     y_pos_list = []
#     speed_list = []
#     acc_list = []
#     collision_list = []
#     t_list = []

#     # Let's implement some simple scenario with all agents
#     c1.set_control(0, 0.5)
#     c2.set_control(0, 2.0)
#     for k in range(int(25/dt)):
#         if k==0:
#             x_pos_list.append([c1.x,c2.x])
#             y_pos_list.append([c1.y,c2.y])
#             speed_list.append([c1.speed,c2.speed])
#             acc_list.append([c1.acceleration,c2.acceleration])
#             collision_list.append(False)
#             t_list.append(k*dt)

#         c2.set_control(*idm_driver(w,c2,c1,idm_configuration["idm_s0"],idm_configuration["idm_T"],idm_configuration["idm_a"],idm_configuration["idm_b"]))
#         # All movable objects will keep their control the same as long as we don't change it.
#         if k*dt >= 13.0:    #after 13s of simulation time 
#             c1.set_control(0,-5.0)
#         w.tick() # This ticks the world for one time step (dt second)
#         # w.render()
#         # time.sleep(dt/16) # Let's watch it 4x

#         collision = w.collision_exists()
#         # if collision: # Or we can check if there is any collision at all.
#         #     print('Collision exists somewhere...')

#         #log all the data
#         x_pos_list.append([c1.x,c2.x])
#         y_pos_list.append([c1.y,c2.y])
#         speed_list.append([c1.speed,c2.speed])
#         acc_list.append([c1.acceleration,c2.acceleration])
#         collision_list.append(collision)
#         t_list.append((k+1)*dt)

#     results_dict = {"x_pos":np.array(x_pos_list).T,
#                     "y_pos":np.array(y_pos_list).T,
#                     "speed":np.array(speed_list).T,
#                     "acc":np.array(acc_list).T,
#                     "collision":np.array(collision_list),
#                     "t":np.array(t_list)}


#     w.close()

#     return results_dict

# def compare_trajectories(results1, results2):
#     """Compare the trajectories of two runs and compute the MSE between the two trajectories (if two different sampling frequency, the higher frequency is downsampled)
#     """
#     run1_x = results1["x_pos"][1]
#     run1_y = results1["y_pos"][1]
#     run1_collision = results1["collision"].astype(np.int32)
#     run1_t = results1["t"]
#     run2_x = results2["x_pos"][1]
#     run2_y = results2["y_pos"][1]
#     run2_collision = results2["collision"].astype(np.int32)
#     run2_t = results2["t"]

#     #downsample
#     if run1_t.shape[0] != run2_t.shape[0]:
#         if run1_t.shape[0] > run2_t.shape[0]:
#             t_common = run2_t
#             run1_x = np.interp(t_common,run1_t,run1_x)
#             run1_y = np.interp(t_common,run1_t,run1_y)
#             run1_collision = np.interp(t_common,run1_t,run1_collision)
#         else:
#             t_common = run1_t
#             run2_x = np.interp(t_common,run2_t,run2_x)
#             run2_y = np.interp(t_common,run2_t,run2_y)
#             run2_collision = np.interp(t_common,run2_t,run2_collision)
         
#     else:
#         t_common = run1_t

#     #compute MSE
#     run1_pos = np.stack([run1_x,run1_y])
#     run2_pos = np.stack([run2_x,run2_y])
#     #MSE for trajectories
#     MSE = (np.linalg.norm(run1_pos-run2_pos,axis=0)**2).mean()
#     #BCE for collision
#     BCE = torch.nn.functional.binary_cross_entropy(torch.Tensor(run1_collision),torch.Tensor(run2_collision))
    
#     return MSE,BCE
# class AXWrapper():
#     def __init__(self) -> None:
#         self.ref_results = run_scenario(0.1,"RK4",0)

#     def ax_wrapper(self, x_dict):
#         integration_method_cost = {"Forward_Euler":1, "RK2":2, "RK4":4}
#         COST= 1/x_dict["dt"] + integration_method_cost[x_dict["integration_method"]] + 1/x_dict["sensor_std"]
#         res = run_scenario(**x_dict)
#         MSE,BCE = compare_trajectories(self.ref_results, res)
#         return {"cost": (MSE, 0.0), "compute_cost":(COST, 0.0) }

# def ax_search():
#     ax_client = AxClient()
#     ax_client.create_experiment(
#     name="carlo_optimization",
#     parameters=[
#         {
#             "name": "dt",
#             "type": "range",
#             "bounds": [0.1, 2.5],
#             "value_type": "float", 
#             "log_scale": False, 
#         },
#         {
#             "name": "integration_method",
#             "type": "choice",
#             "values": ["Forward_Euler", "RK2", "RK4"],
#             "value_type": "str",  # Optional, defaults to inference from type of "bounds".
#             "log_scale": False,  
#         },
#         {
#             "name": "sensor_std",
#             "type": "range",
#             "bounds": [1e-05, 10.0],
#             "value_type": "float", 
#             "log_scale": True,  
#         },
#     ],
#     objective_name="cost",
#     minimize=True,  # Optional, defaults to False.
#     outcome_constraints = ["compute_cost <= 5"]
#     )

#     axw = AXWrapper()
#     for i in range(25):
#         parameters, trial_index = ax_client.get_next_trial()
#         # Local evaluation here can be replaced with deployment to external system.
#         ax_client.complete_trial(trial_index=trial_index, raw_data=axw.ax_wrapper(parameters))
    
#     best_parameters, values = ax_client.get_best_parameters()
#     print(best_parameters)
#     print(values)

# def inner_loop_search(num_runs):
#     failures = []
#     failure_configs = []
#     non_failure_configs = []

#     for i in range(num_runs):
#         """
#         Parameters to search over:
#         ego_max_speed [5,20]
#         ego_max_break_acceleration [-4,-0.5]
#         ego_initial_speed [3,10]
#         idm_T [0.5,3]
#         idm_a [0.5,2.0]
#         idm_b [0.5,2.0]
#         idm_s0 [0,5]
#         """
#         sample_ego_max_speed = np.random.rand()*15 + 5
#         sample_ego_max_break_acceleration = np.random.rand()*3.5 - 4
#         sample_ego_initial_speed = np.random.rand()*7 + 3
#         sample_idm_T = np.random.rand()*1.5 + 0.5
#         sample_idm_a = np.random.rand()*1.5 + 0.5
#         sample_idm_b = np.random.rand()*1.5 + 0.5
#         sample_idm_s0 = np.random.rand()*5

#         config_dict = {"ego_max_speed":sample_ego_max_speed,
#                         "ego_max_break_acceleration":sample_ego_max_break_acceleration,
#                         "ego_initial_speed":sample_ego_initial_speed,
#                         "idm_T":sample_idm_T,
#                         "idm_a":sample_idm_a,
#                         "idm_b":sample_idm_b,
#                         "idm_s0":sample_idm_s0}
        
#         result = run_scenario(0.1,"RK4",0.0,idm_configuration=config_dict)
#         failure = np.any(result["collision"])
#         print(f"Run {i+1}/{num_runs} - Collision found: {failure}")
#         if failure:
#             failures.append(result)
#             failure_configs.append(config_dict)
#         else:
#             non_failure_configs.append(config_dict)
#     print("stop")
#     ego_max_speed_failures = [f["ego_max_speed"] for f in failure_configs]
#     ego_max_speed_non_failures = [f["ego_max_speed"] for f in non_failure_configs]
#     ego_max_break_acceleration_failures = [f["ego_max_break_acceleration"] for f in failure_configs]
#     ego_max_break_acceleration_non_failures = [f["ego_max_break_acceleration"] for f in non_failure_configs]
#     ego_initial_speed_failures = [f["ego_initial_speed"] for f in failure_configs]
#     ego_initial_speed_non_failures = [f["ego_initial_speed"] for f in non_failure_configs]
#     idm_T_failures = [f["idm_T"] for f in failure_configs]
#     idm_T_non_failures = [f["idm_T"] for f in non_failure_configs]
#     idm_a_failures = [f["idm_a"] for f in failure_configs]
#     idm_a_non_failures = [f["idm_a"] for f in non_failure_configs]
#     idm_b_failures = [f["idm_b"] for f in failure_configs]
#     idm_b_non_failures = [f["idm_b"] for f in non_failure_configs]
#     idm_s0_failures = [f["idm_s0"] for f in failure_configs]
#     idm_s0_non_failures = [f["idm_s0"] for f in non_failure_configs]

#     N_bins = 10
#     fig,axs = plt.subplots(1,7,figsize=(25,5))
#     axs[0].hist(ego_max_speed_non_failures,bins=N_bins,color=(0,0.8,0.2),alpha=0.5)
#     axs[0].hist(ego_max_speed_failures,bins=N_bins,color=(1,0,0),alpha=0.5)
#     axs[0].set_title("Ego Max Speed")

#     axs[1].hist(ego_max_break_acceleration_non_failures,bins=N_bins,color=(0,0.8,0.2),alpha=0.5)
#     axs[1].hist(ego_max_break_acceleration_failures,bins=N_bins,color=(1,0,0),alpha=0.5)
#     axs[1].set_title("Ego Max Break Acceleration")

#     axs[2].hist(ego_initial_speed_non_failures,bins=N_bins,color=(0,0.8,0.2),alpha=0.5)
#     axs[2].hist(ego_initial_speed_failures,bins=N_bins,color=(1,0,0),alpha=0.5)
#     axs[2].set_title("Ego Initial Speed")

#     axs[3].hist(idm_T_non_failures,bins=N_bins,color=(0,0.8,0.2),alpha=0.5)
#     axs[3].hist(idm_T_failures,bins=N_bins,color=(1,0,0),alpha=0.5)
#     axs[3].set_title("IDM T")

#     axs[4].hist(idm_a_non_failures,bins=N_bins,color=(0,0.8,0.2),alpha=0.5)
#     axs[4].hist(idm_a_failures,bins=N_bins,color=(1,0,0),alpha=0.5)
#     axs[4].set_title("IDM a")

#     axs[5].hist(idm_b_non_failures,bins=N_bins,color=(0,0.8,0.2),alpha=0.5)
#     axs[5].hist(idm_b_failures,bins=N_bins,color=(1,0,0),alpha=0.5)
#     axs[5].set_title("IDM b")

#     axs[6].hist(idm_s0_non_failures,bins=N_bins,color=(0,0.8,0.2),alpha=0.5)
#     axs[6].hist(idm_s0_failures,bins=N_bins,color=(1,0,0),alpha=0.5)
#     axs[6].set_title("IDM s0")
    
#     fig.tight_layout()
#     fig.savefig("MC_inner_loop_hist.png",dpi=600)
#     fig.show()
#     print("stop")

if __name__=="__main__":
    import matplotlib.pyplot as plt
    # ax_search()
    scenario = SinusoidalCarScenario()
    failure, failure_configs, non_failure_configs = scenario.inner_loop_mc({"dt":0.1,"integration_method":"RK4","sensor_std":0.0},5000,num_processes=8)
    scenario.plot_inner_loop_results(failure_configs,non_failure_configs)
    # scenario.run(scenario.scenario_configuration,{"dt":0.1,"integration_method":"RK4","sensor_std":0.0},render=True)

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




    

