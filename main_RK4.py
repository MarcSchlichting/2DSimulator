import torch
import numpy as np
from tqdm import tqdm
import argparse

from example_stopping_car import StoppingCarScenario
from example_frontal_collision import FrontalCollisionScenario
from example_sinusoidal_car import SinusoidalCarScenario
from example_intersection import OrthogonalIntersectionScenario

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



if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-im",type=str, choices=["Forward_Euler", "RK2", "RK4"])
    # args = parser.parse_args()

    scenarios = [StoppingCarScenario(),OrthogonalIntersectionScenario(),FrontalCollisionScenario(),SinusoidalCarScenario()]
    hf_simulation_config = {"dt":0.1,"integration_method":"RK4","sensor_std":0.0}

    im = "RK4"
    num_samples = 10
    print("INTEGRATION METHOD:",im)
    mse_all = []
    for dt in np.linspace(0.1,1.5,20):
        print("CURRENT DT: ",dt)
        mse_dt = []
        for s_std in np.linspace(0.0,2.0,20):
            print("CURRENT S_STD: ",s_std)
            cf_simulation_config = {"dt":dt,"integration_method":im,"sensor_std":s_std}
            trajectories, scenario_configurations, collisions = evaluate_scenarios(scenarios,num_samples,hf_simulation_config,cf_simulation_config)
            mse, bce = batch_compare_trajectories(trajectories)
            mse_dt.append(mse)
        mse_all.append(mse_dt)
    np.save(f"{im}_all_{num_samples}.npy",np.array(mse_all))
    print("stop")
