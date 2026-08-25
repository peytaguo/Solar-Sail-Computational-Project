import numpy as np

"""
The sun is centred at the origin (0,0,0)
"""

#constants and fixed parameters 
GRAVITATIONAL = 0
SUN_MASS = 0


# input: position vector of the solar sail 
# output: the magnitude distance from the sun (scalar)
def distance_from_sun(position):
    return np.linalg.norm(position)

#input: position vector of solar sail
#outputL acceleration vector from sun on solar sail

# a_g = -GMr/r^3
def a_due_to_gravity(position):
    term = GRAVITATIONAL * SUN_MASS / (distance_from_sun(position))**3 
    return term * position 

# input: the position vector of the solar sail 
# output: the direction the solar sail is travelling in (vector)
def direction_unit_vector(position):
    pass

def 