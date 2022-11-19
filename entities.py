from multiprocessing.sharedctypes import Value
from zlib import Z_FIXED
import numpy as np
from geometry import Point, Rectangle, Circle, Ring
from typing import Union
import copy


class Entity:
    def __init__(self, center: Point, heading: float, movable: bool = True, friction: float = 0, integration_method: str = "Heun"):
        self.center = center # this is x, y
        self.heading = heading
        self.movable = movable
        self.color = 'ghost white'
        self.collidable = True
        self.integration_method = integration_method
        if movable:
            self.friction = friction
            self.velocity = Point(0,0) # this is xp, yp
            self.acceleration = 0 # this is vp (or speedp)
            self.angular_velocity = 0 # this is headingp
            self.inputSteering = 0
            self.inputAcceleration = 0
            self.max_speed = np.inf
            self.min_speed = 0
            self.max_acceleration = np.inf
            self.max_break_acceleration = -np.inf
            self.position_sensor_std = 0
    
    @property
    def speed(self) -> float:
        return self.velocity.norm(p = 2) if self.movable else 0
    
    def set_control(self, inputSteering: float, inputAcceleration: float):
        self.inputSteering = inputSteering
        self.inputAcceleration = inputAcceleration
    
    @property
    def rear_dist(self) -> float: # distance between the rear wheels and the center of mass. This is needed to implement the kinematic bicycle model dynamics
        if isinstance(self, RectangleEntity):
            # only for this function, we assume
            # (i) the longer side of the rectangle is always the nominal direction of the car
            # (ii) the center of mass is the same as the geometric center of the RectangleEntity.
            return np.maximum(self.size.x, self.size.y) / 2.
        elif isinstance(self, CircleEntity):
            return self.radius
        elif isinstance(self, RingEntity):
            return (self.inner_radius + self.outer_radius) / 2.
        raise NotImplementedError
    
    def tick(self, dt: float):
        if self.movable:
            if self.integration_method == "RK4":
                def f(z,acc):
                    return np.array([z[2],z[3],acc[0],acc[1]])

                if np.abs(self.inputSteering) > 0.0:
                    raise ValueError
                
                heading = self.heading
                velocity_rframe = np.array([np.cos(heading) * self.speed, np.sin(heading) * self.speed])
                acc_abs = np.clip(self.inputAcceleration - self.friction * self.speed, self.max_break_acceleration, self.max_acceleration)
                acc_new = np.array([np.cos(heading) * acc_abs, np.sin(heading) * acc_abs])
                z_current = np.array([self.center.x,self.center.y,velocity_rframe[0],velocity_rframe[1]])

                #RK4 
                k1 = f(z_current,acc_new)
                k2 = f(z_current + dt * k1 / 2, acc_new)
                k3 = f(z_current + dt * k2 / 2, acc_new)
                k4 = f(z_current + dt * k3, acc_new)

                z_new = z_current + dt/6 * (k1 + 2*k2 + 2*k3 +k4)

                #check for negative speed
                # new_speed = np.dot(np.array([np.cos(heading),np.sin(heading)]),z_new[2:])

                #new condition, speed small and negative acc
                # prelim_velocity = Point(np.cos(-self.heading) * z_new[2],-np.sin(self.heading) * z_new[3])
                # prelim_speed = np.sqrt(prelim_velocity.x**2 + prelim_velocity.y**2)
                preliminary_heading = np.mod(np.arctan2(z_new[3],z_new[2]),2 * np.pi)
                if np.abs(preliminary_heading-self.heading) > 0.5:
                    z_new[:2] = np.array([self.center.x, self.center.y])
                    z_new[2:] = np.zeros((2,))
                    self.heading = heading

                # if z_new[0]>self.center.x:
                #     z_new[:2] = np.array([self.center.x, self.center.y])
                #     z_new[2:] = np.zeros((2,))
                #     self.heading = heading
                else:
                    self.heading = np.mod(np.arctan2(z_new[3],z_new[2]),2 * np.pi)

                self.center = Point(z_new[0],z_new[1])
                self.velocity = Point(np.cos(-self.heading) * z_new[2],-np.sin(self.heading) * z_new[3]) # in body frame
                
                self.acceleration = acc_abs
                self.angular_velocity = self.speed * self.inputSteering

                self.buildGeometry()

            elif self.integration_method == "RK2":
                def f(z,acc):
                    return np.array([z[2],z[3],acc[0],acc[1]])

                if np.abs(self.inputSteering) > 0.0:
                    raise ValueError
                
                heading = self.heading
                velocity_rframe = np.array([np.cos(heading) * self.speed, np.sin(heading) * self.speed])
                acc_abs = np.clip(self.inputAcceleration - self.friction * self.speed, self.max_break_acceleration, self.max_acceleration)
                acc_new = np.array([np.cos(heading) * acc_abs, np.sin(heading) * acc_abs])
                z_current = np.array([self.center.x,self.center.y,velocity_rframe[0],velocity_rframe[1]])

                #RK2
                k1 = dt* f(z_current, acc_new)
                k2 = dt *f(z_current + k1, acc_new)

                z_new = z_current + (k1 + k2)/2

                #check for negative speed
                # new_speed = np.dot(np.array([np.cos(heading),np.sin(heading)]),z_new[2:])
                
                if z_new[0]>self.center.x:
                    z_new[:2] = np.array([self.center.x, self.center.y])
                    z_new[2:] = np.zeros((2,))
                    self.heading = heading
                else:
                    self.heading = np.mod(np.arctan2(z_new[3],z_new[2]),2 * np.pi)

                self.center = Point(z_new[0],z_new[1])
                self.velocity = Point(np.cos(-self.heading) * z_new[2],-np.sin(self.heading) * z_new[3]) # in body frame

                self.center = Point(z_new[0],z_new[1])
                self.heading = np.mod(np.arctan2(z_new[3],z_new[2]),2 * np.pi)
                self.velocity = Point(np.cos(-self.heading) * z_new[2],-np.sin(self.heading) * z_new[3]) # in body frame
                
                self.acceleration = acc_abs
                self.angular_velocity = self.speed * self.inputSteering

                self.buildGeometry()
            
            elif self.integration_method == "RK1":
                def f(z,acc):
                    return np.array([z[2],z[3],acc[0],acc[1]])

                if np.abs(self.inputSteering) > 0.0:
                    raise ValueError
                
                heading = self.heading
                velocity_rframe = np.array([np.cos(heading) * self.speed, np.sin(heading) * self.speed])
                acc_abs = np.clip(self.inputAcceleration - self.friction * self.speed, self.max_break_acceleration, self.max_acceleration)
                acc_new = np.array([np.cos(heading) * acc_abs, np.sin(heading) * acc_abs])
                z_current = np.array([self.center.x,self.center.y,velocity_rframe[0],velocity_rframe[1]])

                #RK2
                k1 = f(z_current, acc_new)
            
                z_new = z_current + dt * k1

                #check for negative speed
                # new_speed = np.dot(np.array([np.cos(heading),np.sin(heading)]),z_new[2:])
                
                if z_new[0]>self.center.x:
                    z_new[:2] = np.array([self.center.x, self.center.y])
                    z_new[2:] = np.zeros((2,))
                    self.heading = heading
                else:
                    self.heading = np.mod(np.arctan2(z_new[3],z_new[2]),2 * np.pi)

                self.center = Point(z_new[0],z_new[1])
                self.velocity = Point(np.cos(-self.heading) * z_new[2],-np.sin(self.heading) * z_new[3]) # in body frame

                self.center = Point(z_new[0],z_new[1])
                self.heading = np.mod(np.arctan2(z_new[3],z_new[2]),2 * np.pi)
                self.velocity = Point(np.cos(-self.heading) * z_new[2],-np.sin(self.heading) * z_new[3]) # in body frame
                
                self.acceleration = acc_abs
                self.angular_velocity = self.speed * self.inputSteering

                self.buildGeometry()

                # print("stop")

            # elif self.integration_method == "Heun":
            #     # print("Heun")
            #     speed = self.speed
            #     heading = self.heading
            

            #     # Point-mass dynamics based on
            #     # "Active Preference-Based Learning of Reward Functions" by
            #     # Dorsa Sadigh, Anca D. Dragan, S. Shankar Sastry, Sanjit A. Seshia
                
            #     new_angular_velocity = speed * self.inputSteering
            #     new_acceleration = np.clip(self.inputAcceleration - self.friction * speed, self.max_break_acceleration, self.max_acceleration)
                
            #     new_heading = heading + (self.angular_velocity + new_angular_velocity) * dt / 2.
            #     new_speed = np.clip(speed + (self.acceleration + new_acceleration) * dt / 2., self.min_speed, self.max_speed)
                
            #     new_velocity = Point(((speed + new_speed) / 2.) * np.cos((new_heading + heading) / 2.),
            #                             ((speed + new_speed) / 2.) * np.sin((new_heading + heading) / 2.))
                
            #     new_center = self.center + (self.velocity + new_velocity) * dt / 2.
                
                
                
            #     self.center = new_center
            #     self.heading = np.mod(new_heading, 2*np.pi) # wrap the heading angle between 0 and +2pi
            #     self.velocity = new_velocity
            #     self.acceleration = new_acceleration
            #     self.angular_velocity = new_angular_velocity
                
            #     self.buildGeometry()

            elif self.integration_method == "Forward_Euler":
                # print("Forward Euler")
                speed = self.speed
                heading = self.heading
            

                # Point-mass dynamics based on
                # "Active Preference-Based Learning of Reward Functions" by
                # Dorsa Sadigh, Anca D. Dragan, S. Shankar Sastry, Sanjit A. Seshia
                
                new_angular_velocity = speed * self.inputSteering
                new_acceleration = np.clip(self.inputAcceleration - self.friction * speed, self.max_break_acceleration, self.max_acceleration)
                
                new_heading = heading + new_angular_velocity * dt 
                new_speed = np.clip(speed + new_acceleration * dt , self.min_speed, self.max_speed)
                
                new_velocity = Point((new_speed * np.cos(new_heading)), (new_speed * np.sin(new_heading)))
                
                new_center = self.center + new_velocity * dt 
                
                
                
                self.center = new_center
                self.heading = np.mod(new_heading, 2*np.pi) # wrap the heading angle between 0 and +2pi
                self.velocity = new_velocity
                self.acceleration = new_acceleration
                self.angular_velocity = new_angular_velocity
                
                self.buildGeometry()
            
            else:   #this defaults to the bicycle kinematics
                raise ValueError
                speed = self.speed
                heading = self.heading
            
                # Kinematic bicycle model dynamics based on
                # "Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design" by
                # Jason Kong, Mark Pfeiffer, Georg Schildbach, Francesco Borrelli
                lr = self.rear_dist
                lf = lr # we assume the center of mass is the same as the geometric center of the entity
                beta = np.arctan(lr / (lf + lr) * np.tan(self.inputSteering))
                
                new_angular_velocity = speed * self.inputSteering # this is not needed and used for this model, but let's keep it for consistency (and to avoid if-else statements)
                new_acceleration =  np.clip(self.inputAcceleration - self.friction * speed, self.max_break_acceleration, self.max_acceleration)
                new_speed = np.clip(speed + new_acceleration * dt, self.min_speed, self.max_speed)
                new_heading = heading + ((speed + new_speed)/lr)*np.sin(beta)*dt/2.
                angle = (heading + new_heading)/2. + beta
                new_center = self.center + (speed + new_speed)*Point(np.cos(angle), np.sin(angle))*dt / 2.
                new_velocity = Point(new_speed * np.cos(new_heading), new_speed * np.sin(new_heading))
                
                '''
                # Point-mass dynamics based on
                # "Active Preference-Based Learning of Reward Functions" by
                # Dorsa Sadigh, Anca D. Dragan, S. Shankar Sastry, Sanjit A. Seshia
                
                new_angular_velocity = speed * self.inputSteering
                new_acceleration = self.inputAcceleration - self.friction * speed
                
                new_heading = heading + (self.angular_velocity + new_angular_velocity) * dt / 2.
                new_speed = np.clip(speed + (self.acceleration + new_acceleration) * dt / 2., self.min_speed, self.max_speed)
                
                new_velocity = Point(((speed + new_speed) / 2.) * np.cos((new_heading + heading) / 2.),
                                        ((speed + new_speed) / 2.) * np.sin((new_heading + heading) / 2.))
                
                new_center = self.center + (self.velocity + new_velocity) * dt / 2.
                
                '''
                
                self.center = new_center
                self.heading = np.mod(new_heading, 2*np.pi) # wrap the heading angle between 0 and +2pi
                self.velocity = new_velocity
                self.acceleration = new_acceleration
                self.angular_velocity = new_angular_velocity
                
                self.buildGeometry()
    
    def buildGeometry(self): # builds the obj
        raise NotImplementedError
        
    def collidesWith(self, other: Union['Point','Entity']) -> bool:
        if isinstance(other, Entity):
            return self.obj.intersectsWith(other.obj)
        elif isinstance(other, Point):
            return self.obj.intersectsWith(other)
        raise NotImplementedError
        
    def distanceTo(self, other: Union['Point','Entity']) -> float:
        if isinstance(other, Entity):
            return self.obj.distanceTo(other.obj)
        elif isinstance(other, Point):
            return self.obj.distanceTo(other)
        raise NotImplementedError
        
    def copy(self):
        return copy.deepcopy(self)
        
    @property
    def x(self):
        return self.center.x

    @property
    def y(self):
        return self.center.y
        
    @property
    def xp(self):
        return self.velocity.x

    @property
    def yp(self):
        return self.velocity.y
    
class RectangleEntity(Entity):
    def __init__(self, center: Point, heading: float, size: Point, movable: bool = True, friction: float = 0):
        super(RectangleEntity, self).__init__(center, heading, movable, friction)
        self.size = size
        self.buildGeometry()
    
    @property
    def edge_centers(self):
        edge_centers = np.zeros((4,2), dtype=np.float32)
        x = self.center.x
        y = self.center.y
        w = self.size.x
        h = self.size.y
        edge_centers[0] = [x + w / 2. * np.cos(self.heading), y + w / 2. * np.sin(self.heading)]
        edge_centers[1] = [x - h / 2. * np.sin(self.heading), y + h / 2. * np.cos(self.heading)]
        edge_centers[2] = [x - w / 2. * np.cos(self.heading), y - w / 2. * np.sin(self.heading)]
        edge_centers[3] = [x + h / 2. * np.sin(self.heading), y - h / 2. * np.cos(self.heading)]
        return edge_centers
        
    @property
    def corners(self):
        ec = self.edge_centers
        c = np.array([self.center.x, self.center.y])
        corners = []
        corners.append(Point(*(ec[1] + ec[0] - c)))
        corners.append(Point(*(ec[2] + ec[1] - c)))
        corners.append(Point(*(ec[3] + ec[2] - c)))
        corners.append(Point(*(ec[0] + ec[3] - c)))
        return corners
        
    def buildGeometry(self):
        C = self.corners
        self.obj = Rectangle(*C[:-1])
        
class CircleEntity(Entity):
    def __init__(self, center: Point, heading: float, radius: float, movable: bool = True, friction: float = 0):
        super(CircleEntity, self).__init__(center, heading, movable, friction)
        self.radius = radius
        self.buildGeometry()
        
    def buildGeometry(self):
        self.obj = Circle(self.center, self.radius)
                    
class RingEntity(Entity):
    def __init__(self, center: Point, heading: float, inner_radius: float, outer_radius: float, movable: bool = True, friction: float = 0):
        super(RingEntity, self).__init__(center, heading, movable, friction)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.buildGeometry()
        
    def buildGeometry(self):
        self.obj = Ring(self.center, self.inner_radius, self.outer_radius)