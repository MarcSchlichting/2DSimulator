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

class OrthogonalIntersectionScenario(object):
    def __init__(self) -> None:
        #default configuration for inner loop
        self.name = "OrthogonalIntersectionScenario"
        self.scenario_configuration = {"ego_max_speed":10,
                                    "ego_max_break_acceleration":-5.8,
                                    "ego_initial_speed":3.0,
                                    "other_car_initial_speed":3.0,
                                    "other_car_acc":0.05,
                                    "idm_T":1.0,
                                    "idm_a":1.5,
                                    "idm_b":1.5,
                                    "idm_s0":1.1}

        #define the type for each of the variables
        self.scenario_configuration_domain = {"ego_max_speed":{"type":"range", "values":[5,20]},
                                    "ego_max_break_acceleration":{"type":"range", "values":[-4,-0.5]},
                                    "ego_initial_speed":{"type":"range", "values":[3,10]},
                                    "other_car_initial_speed":{"type":"range","values":[1.0,5.0]},
                                    "other_car_acc":{"type":"range","values":[0.0,0.5]},
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

        w.add(Painting(Point(95, 107.5), Point(74, 29), 'gray80')) # We build a sidewalk.
        w.add(Painting(Point(17.5, 107.5), Point(59, 29), 'gray80')) # We build a sidewalk.
        w.add(RectangleBuilding(Point(95, 107.5), Point(70, 25))) 
        w.add(RectangleBuilding(Point(17.5, 107.5), Point(55, 25))) 

        w.add(Painting(Point(95, 40), Point(74,84), 'gray80'))
        w.add(Painting(Point(17.5, 40), Point(59,84), 'gray80'))
        w.add(RectangleBuilding(Point(95, 40), Point(70, 80))) 
        w.add(RectangleBuilding(Point(17.5, 40), Point(55, 80)))



        # The other car
        c1= Car(Point(50,60), np.pi/2, 'blue')
        c1.velocity = Point(scenario_configuration["other_car_initial_speed"],0.0) 
        c1.integration_method = simulation_configuration["integration_method"]
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
            
            # if math.sin(2*math.pi/scenario_configuration["lead_car_osc_period"]*k*simulation_configuration["dt"]) > 0:
            #     c1_control = (0,scenario_configuration["lead_car_acc"])
            # else:
            #     c1_control = (0,scenario_configuration["lead_car_break_acc"])
            

            c1.set_control(0.0,scenario_configuration["other_car_acc"])
            
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



        # with WorkerPool(n_jobs=num_processes) as pool:

        #     for result in pool.imap(self.inner_loop_one_step, num_trials * [(simulation_configuration,)],progress_bar=True):
        #     # result = self.inner_loop_one_step()

        #         if result[0] is not None:
        #             failures.append(result[0])
        #         if result[1] is not None:
        #             failure_configs.append(result[1])
        #         if result[2] is not None:
        #             non_failure_configs.append(result[2])
        

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
        fig.savefig(f"MC_inner_loop_hist_intersection_{int(time.time())}.png",dpi=600)
        # fig.show()
        print("stop")













if __name__=="__main__":
    import matplotlib.pyplot as plt
    # ax_search()
    scenario = OrthogonalIntersectionScenario()
    # failure, failure_configs, non_failure_configs = scenario.inner_loop_mc({"dt":0.1,"integration_method":"RK4","sensor_std":0.0},5000,num_processes=8)
    # scenario.plot_inner_loop_results(failure_configs,non_failure_configs)
    scenario.run(scenario.scenario_configuration,{"dt":0.1,"integration_method":"RK4","sensor_std":0.0},render=True)

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




    

