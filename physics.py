import numpy as np

"""
The sun is centred at the origin (0,0,0)
"""

#constants and fixed parameters 
GRAVITATIONAL = 6.6743e-11 #m^3 kg^-1 s^-2
SUN_MASS = 1.989e30 #kg
SAIL_AREA = 10 #m^2



# input: position vector of the solar sail 
# output: the magnitude distance from the sun (scalar)
def distance_from_sun(position):
    return np.linalg.norm(position)

#input: position vector of solar sail
#outputL acceleration vector from sun on solar sail

# a_g = -GMr/r^3
def a_due_to_gravity(position):
    scalar_term = GRAVITATIONAL * SUN_MASS / (distance_from_sun(position))**3 
    return scalar_term * position 

# input: the position vector of the solar sail 
# output: the direction the solar sail is travelling in (vector)
def direction_unit_vector(position):
    return np.linalg.norm(position)

def 