import torch
import numpy as np
from tqdm import tqdm
from ax.service.ax_client import AxClient
import pandas as pd


from example_stopping_car import StoppingCarScenario
from example_frontal_collision import FrontalCollisionScenario
from example_sinusoidal_car import SinusoidalCarScenario
from example_intersection import OrthogonalIntersectionScenario

scenarios = [StoppingCarScenario(),OrthogonalIntersectionScenario(),FrontalCollisionScenario(),SinusoidalCarScenario()]
hf_simulation_config = {"dt":0.1,"integration_method":"RK4","sensor_std":0.0}
compute_budget = 100
rollouts_per_scenario = 10
num_iterations = 200

def evaluate_scenarios(scenarios:list,num_per_scenario:int,hf_simulation_configuration:dict,cf_simulation_configuration):
    trajectories = []   #list of tuples (hf_trajectory,cf_trajectory)
    scenario_configurations = [] #list of scenario configurations that lead to a failure
    collisions = []     #list of tuples (collision_hf, collision_cf)
    
    for s in tqdm(scenarios):
        for i in range(num_per_scenario):
            phi_sample = s.sample_scenario_configuration()
            results_hf = s.run(phi_sample,hf_simulation_configuration)
            results_cf = s.run(phi_sample,cf_simulation_configuration)
            collision_hf = np.any(results_hf["collision"])
            collision_cf = np.any(results_cf["collision"])
            trajectories.append((results_hf,results_cf))
            scenario_configurations.append(phi_sample)
            collisions.append((collision_hf,collision_cf))

    return trajectories, scenario_configurations, collisions

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

def batch_compare_trajectories(trajectories):
    mse_list = []
    bce_list = []

    for trajectory1, trajectory2 in trajectories:
        mse,bce = compare_trajectories(trajectory1,trajectory2)
        mse_list.append(mse)
        bce_list.append(bce)
    
    return mse_list, bce_list

def objective(cf_simulation_config):
    trajectories, scenario_configurations, collisions = evaluate_scenarios(scenarios,rollouts_per_scenario,hf_simulation_config,cf_simulation_config)
    mse, bce = batch_compare_trajectories(trajectories)
    mse = torch.mean(torch.Tensor(mse))

    if cf_simulation_config["integration_method"]=="RK1":
        int_cost = 1
    elif cf_simulation_config["integration_method"]=="RK2":
        int_cost = 2
    elif cf_simulation_config["integration_method"]=="RK4":
        int_cost = 4
    else:
        raise ValueError
    compute_cost = 25/cf_simulation_config["dt"]*(int_cost + 1/cf_simulation_config["sensor_std"])
    if compute_cost>compute_budget:
        mse += 10000
    
    return {"mse":(mse.item(),0.0)}

if __name__=="__main__":
    ax_client = AxClient()
    # ax_client = AxClient(enforce_sequential_optimization=False)

    ax_client.create_experiment(
        name="2d_sim_fidelity",
        parameters=[
            {
                "name": "dt",
                "type": "range",
                "bounds": [0.1, 1.5],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, defaults to False.
            },
            {
                "name": "integration_method",
                "type": "choice",
                "values": ["RK1","RK2","RK4"],
            },
            {
                "name": "sensor_std",
                "type": "range",
                "bounds": [0.01, 2.0],
            },
        ],
        objective_name="mse",
        minimize=True,  
    )

    for i in range(num_iterations):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        raw_data = objective(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        ax_client.generation_strategy.trials_as_df.to_csv("meta_learning_200.csv")
        best_parameters, values = ax_client.get_best_parameters()
        print("Best Parameters: ",best_parameters)
    
    best_parameters, values = ax_client.get_best_parameters()

    print(ax_client.generation_strategy.trials_as_df)
    print("Best Parameters: ",best_parameters)
    print("Objective Value: ",values[0])

    