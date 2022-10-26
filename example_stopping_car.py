import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
from idm import idm_driver

human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
# A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
# For both of these objects, we give the center point and the size.
w.add(Painting(Point(60, 106.5), Point(120, 27), 'gray80')) # We build a sidewalk.
w.add(RectangleBuilding(Point(60, 107.5), Point(120, 25))) # The RectangleBuilding is then on top of the sidewalk, with some margin.

w.add(Painting(Point(60, 41), Point(120, 82), 'gray80')) # We build a sidewalk.
w.add(RectangleBuilding(Point(60, 40), Point(120, 80))) # The RectangleBuilding is then on top of the sidewalk, with some margin.


# # Let's repeat this for 4 different RectangleBuildings.
# w.add(Painting(Point(8.5, 106.5), Point(17, 27), 'gray80'))
# w.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))

# w.add(Painting(Point(8.5, 41), Point(17, 82), 'gray80'))
# w.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))

# w.add(Painting(Point(71.5, 41), Point(97, 82), 'gray80'))
# w.add(RectangleBuilding(Point(72.5, 40), Point(95, 80)))



# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1= Car(Point(80,90), np.pi, 'blue')
c1.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
w.add(c1)

c2 = Car(Point(110,90), np.pi)
# c2.driver_model = "IDM"
c2.max_speed = 10
c2.velocity = Point(3.0,0.)

w.add(c2)


w.render() # This visualizes the world we just constructed.


if not human_controller:
    # Let's implement some simple scenario with all agents
    c1.set_control(0, 0.1)
    c2.set_control(0, 2.0)
    for k in range(200):
        c2.set_control(*idm_driver(w,c2,c1))
        # All movable objects will keep their control the same as long as we don't change it.
        if k == 130: # Let's say the first Car will release throttle (and start slowing down due to friction)
            c1.set_control(0,-5.0)
            print("breaking")
        w.tick() # This ticks the world for one time step (dt second)
        w.render()
        time.sleep(dt/8) # Let's watch it 4x
    

        if w.collision_exists(): # Or we can check if there is any collision at all.
            print('Collision exists somewhere...')
    w.close()
