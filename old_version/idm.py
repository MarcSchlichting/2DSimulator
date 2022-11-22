import numpy as np
import agents

def idm_driver(w,car_ego,car1,s_0,T,a,b):
    #Model Parameters
    v_0 = car_ego.max_speed
    s_0 =s_0 #1.1
    T =T #1.0
    a =a #1.5
    b =b #1.5
    delta = 4

    #Helper Quantities
    v_alpha = car_ego.speed
    delta_v_alpha = np.abs(car_ego.speed-car1.speed)
    pos_diff = car_ego.center - car1.center 
    diff_norm = np.linalg.norm(np.array([pos_diff.x,pos_diff.y])) + car_ego.position_sensor_std * np.random.randn()
    s_alpha = diff_norm - car_ego.size.x/2 - car1.size.x /2
    s_star = s_0 + v_alpha * T + (v_alpha * delta_v_alpha)/(2 * np.sqrt(a * b))

    #compute v_dot
    v_dot = a * (1- (v_alpha/v_0)**delta - (s_star/s_alpha)**2)

    return 0, v_dot
