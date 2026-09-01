import numpy as np
import math

"""
The sun is centred at the origin (0,0,0)
"""

#constants and fixed parameters 
GRAVITATIONAL = 6.6743e-11 #m^3 kg^-1 s^-2
SUN_MASS = 1.989e30 #kg
SAIL_AREA = 10 #m^2
SUN_IRRADIANCE = 1361 # watt/m^2 at 1AU
SPEED_LIGHT = 299792458 #m/s
R = 1 #the fraction of photons that undergo specular reflection
AU = 1.496e11 #distance of Earth from Sun in metres 

position_vector = np.array( (AU,0,0) )
velocity_vector = np.array( (0,0,0) )

a_angle = 0 #angle between sail normal and sunlight
b_angle = 0 #which way around the sail is tilt

# input: p
# output: the magnitude distance from the sun (scalar)
def distance_from_sun():
    return np.linalg.norm(position_vector)

#input: position vector of solar sail
#outputL acceleration vector from sun on solar sail
# a_g = -GMr/r^3
def a_due_to_gravity():
    scalar_term = GRAVITATIONAL * SUN_MASS / ( distance_from_sun() ) ** 3 
    return -1*scalar_term * position_vector 

# input: the position vector of the solar sail 
# output: the direction vector of photon reaching solar sail
def sun_unit_vector():
    return position_vector / np.linalg.norm(position_vector)

#input: the velocity vector of the solar sail
#output: the vector perpendicular to sun unit vector and perpendicular to the orbital plane 
def N_unit_vector():
    cross_product = np.cross(sun_unit_vector(), velocity_vector)

    if np.linalg.norm(cross_product) < 1e-12:
        raise ValueError(
            "Orbital plane undefined: velocity is parallel to radial direction."
        )

    return cross_product / np.linalg.norm(cross_product)

#input: the position and velocity vector of the solar sail
#output: the vector perpendicular to sun unit vector in the orbital plane 
def T_unit_vector():
    n = N_unit_vector()
    s = sun_unit_vector()
    return np.cross(n,s)

#input: the angle of tilt of the solar sail 
#output: the normal vector of the solar sail
def solar_sail_normal_vector():
    s = sun_unit_vector()
    t = T_unit_vector()
    n = N_unit_vector()

    s_component = math.cos(a_angle)
    t_component = math.sin(a_angle) * math.cos(b_angle)
    n_component = math.sin(a_angle) * math.sin(b_angle)

    return s_component * s + t_component * t + n_component * n 

# input: position of solar sail 
# output: the radiant power per unit area hitting the solar sail at given position (scalar)
def irradiance():
    distance = distance_from_sun()
    return SUN_IRRADIANCE * (AU/distance)**2

#input: position of solar sail
# output: the momentum imparted on the solar sail per second
def incoming_momentum_ps():
    current_irradiance = irradiance()
    a = max(0, np.dot(sun_unit_vector(), solar_sail_normal_vector()))

    #depending on the tilt of the solar sail, flux decreases if tilt is not normal to incoming photons. 
    return current_irradiance * SAIL_AREA * a * sun_unit_vector() / SPEED_LIGHT 

#input: position of solar sail, velocity_vector
#output: the force imparted on the solar sail due to specular reflection
def dF_reflection():
    a = max(0, np.dot(sun_unit_vector(), solar_sail_normal_vector()))
    return R * 2 * irradiance() * SAIL_AREA / SPEED_LIGHT * a**2 * solar_sail_normal_vector()

#input: 
#output: the force on the the solar sail due to absorption of photons
def dF_absorbed():
    return (1-R)*incoming_momentum_ps()

#input:
#output: the total force on the solar sail due to radiation pressure (vector)
def total_force_radiation():
    return dF_reflection() + dF_absorbed()


# Testing 
position_vector = np.array([AU, 0.0, 0.0])
velocity_vector = np.array([0.0, 29780.0, 0.0])

a_angle = 0
b_angle = 0

print(irradiance())
print(total_force_radiation())