import torch
import numpy as np
from tqdm import tqdm
from ax.service.ax_client import AxClient
import pandas as pd
import math
import pickle


# from example_diagonal_intersection import DiagonalIntersectionScenario
from example_intersection2 import OrthogonalIntersectionScenario2
from example_parallel_collision import ParallelCollisionScenario

scenarios = [OrthogonalIntersectionScenario2(),ParallelCollisionScenario()]
hf_simulation_config = {"dt":0.1,"integration_method":"RK4","sensor_std":0.1}
rollouts_per_scenario = 10000

def evaluate_scenarios(scenarios:list,num_per_scenario:int,hf_simulation_configuration:dict,cf_simulation_configuration:dict):
    trajectories = []   #list of trajectories that lead to failure
    scenario_parameters = [] #list of scenario configurations that lead to a failure
    mse_list = []
    case_condition_list = []
    scenario_list = [] #list of name of scenario associated with failure
    
    for s in tqdm(scenarios):
        for i in range(num_per_scenario):
            phi_sample = s.sample_scenario_configuration()
            results_cf = s.run(phi_sample,cf_simulation_configuration)
            collision_cf = np.any(results_cf["collision"])
            if collision_cf:
                results_hf = s.run(phi_sample,hf_simulation_configuration)
                collision_hf = np.any(results_hf["collision"])
                mse,bce = compare_trajectories(results_hf, results_cf)
                if (collision_hf==True) & (collision_cf==True):
                    case_condition = "TP"
                # elif collision_hf==False & collision_cf==False:
                #     case_condition = "TN"
                # elif collision_hf==True & collision_cf==False:
                #     case_condition = "FN"
                elif (collision_hf==False) & (collision_cf==True):
                    case_condition = "FP"
                trajectories.append((results_hf,results_cf))
                scenario_parameters.append(phi_sample)
                mse_list.append(mse)
                case_condition_list.append(case_condition)
                scenario_list.append(s.name)

    return trajectories, scenario_parameters, mse_list, case_condition_list, scenario_list

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

if __name__=="__main__":

    cf_simulation_config = {"dt":0.2961962,"integration_method":"RK2","sensor_std":0.46666667}

    trajectories, scenario_parameters, mses, case_conditions, scenario_names = evaluate_scenarios(scenarios,rollouts_per_scenario,hf_simulation_config,cf_simulation_config)
    total_failures = len(case_conditions)
    fraction_real_failures = case_conditions.count("TP")/(rollouts_per_scenario*len(scenarios))
    precision = case_conditions.count("TP") / (case_conditions.count("TP") + case_conditions.count("FP"))

    compute_cost = 25/cf_simulation_config["dt"]*(float(cf_simulation_config["integration_method"][-1])+1/cf_simulation_config["sensor_std"])
    print(f"Compute Cost: {compute_cost}")
    print(f"Total Evaluations: {rollouts_per_scenario*len(scenarios)}")
    print(f"Total Failures Found (FP + TP): {total_failures}")
    print(f"Actual Failures: {case_conditions.count('TP')}")
    print(f"Fraction Real Failures: {fraction_real_failures}")
    print(f"Precision: {precision}")
        


        



# for c in case_conditions:
#     print(c)

    