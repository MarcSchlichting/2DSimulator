import torch
import numpy as np
from tqdm import tqdm
from ax.service.ax_client import AxClient
import pandas as pd
import math


# from example_diagonal_intersection import DiagonalIntersectionScenario
from example_intersection2 import OrthogonalIntersectionScenario2
from example_parallel_collision import ParallelCollisionScenario

scenarios = [OrthogonalIntersectionScenario2(),ParallelCollisionScenario()]
hf_simulation_config = {"dt":0.1,"integration_method":"RK4","sensor_std":0.1}
rollouts_per_scenario = 500

def evaluate_scenarios(scenarios:list,num_per_scenario:int,hf_simulation_configuration:dict,cf_simulation_configuration):
    trajectories = []   #list of tuples (hf_trajectory,cf_trajectory)
    scenario_configurations = [] #list of scenario configurations that lead to a failure
    collisions = []     #list of tuples (collision_hf, collision_cf)
    
    for s in tqdm(scenarios):
        for i in tqdm(range(num_per_scenario)):
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

def calculate_confusion_matrix(collisions):
    """collisions is a list of tuples [(collision_hf, collision_cf)]"""
    collisions = np.array(collisions)
    TP = np.sum(np.logical_and(collisions[:,0]==collisions[:,1], collisions[:,0]==True))
    TN = np.sum(np.logical_and(collisions[:,0]==collisions[:,1], collisions[:,0]==False))
    FP = np.sum(np.logical_and(collisions[:,0]!=collisions[:,1], collisions[:,0]==False))
    FN = np.sum(np.logical_and(collisions[:,0]!=collisions[:,1], collisions[:,0]==True))

    return TP,TN,FP,FN



if __name__=="__main__":
    best_parameters_file = "meta_training_700_sem_best_parameters.csv"
    df_best_parameters = pd.read_csv(best_parameters_file)
    best_integration_method =  list(df_best_parameters["integration_method"])[-1]
    best_dt = list(df_best_parameters["dt"])[-1]
    best_sensor_std = list(df_best_parameters["sensor_std"])[-1]
    cf_simulation_config = {"dt":best_dt,"sensor_std":best_sensor_std,"integration_method":best_integration_method}
    trajectories, scenario_configurations, collisions = evaluate_scenarios(scenarios,rollouts_per_scenario,hf_simulation_config,cf_simulation_config)
    TP,TN,FP,FN = calculate_confusion_matrix(collisions)
    miss_rate = FN/(TP+FN)
    false_positive_rate= FP/(FP+TN)
    mse_list,bce_list = batch_compare_trajectories(trajectories)
    mse_mean = torch.mean(torch.Tensor(mse_list)).item()
    mse_std = torch.std(torch.Tensor(mse_list)).item()
    print("Mean MSE: ",mse_mean)
    print("Std MSE: ",mse_std)
    print("Miss Rate: ",miss_rate)
    print("False Positive Rate", false_positive_rate)


    