import numpy as np
import math

"""
The sun is centred at the origin (0,0,0)
"""

#constants and fixed parameters 
GRAVITATIONAL = 6.6743e-11 #m^3 kg^-1 s^-2
SPEED_LIGHT = 299792458 #m/s
AU = 1.496e11 #distance of Earth from Sun in metres 
DT = 1 #timestep

SUN_IRRADIANCE = 1361 # watt/m^2 at 1AU
SUN_MASS = 1.989e30 #kg
SUN_RADIUS = 6.957e8 #in m

MARS_DISTANCE = 1.524*AU
MARS_MASS = 6.417e23 #in kg
MARS_RADIUS = 3389.5e3 # in m
MARS_HILL_RADIUS = MARS_DISTANCE * (MARS_MASS / SUN_MASS) ** (1/3)

SAIL_AREA = 80 #m^2 #80m^2 approximately using NASA's solar sail from: https://www.nasa.gov/smallspacecraft/what-is-acs3/ 
MASS = 16 #mass of spacecraft kg from: https://ntrs.nasa.gov/api/citations/20230008378/downloads/Wilkie_ACS3_mission_update_ISSS_2023_20230530_rev_a.pdf 
R = 0.82 #the fraction of photons that undergo specular reflection

position_vector = np.array( (AU,0,0) ) #position vector of spacecraft from the sun
velocity_vector = np.array( (0,0,0) ) #velocity of spacecraft


a_angle = 0 #angle between sail normal and sunlight
b_angle = 0 #which way around the sail is tilt

#important parameters of the solar sail combined into one vector 
solar_sail_state = np.array(
    [AU, 0, 0, 0, 0, 0, 0],
    dtype=float
)  # x, y, z, vx, vy, vz, t


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

def total_acceleration():
    return a_due_to_gravity() + total_force_radiation() / MASS

#calculate both velocity and position vectors first before updating!

#########################################################################################
#using Euler integration, this will end up having significant numerical errors, 
#will fix later
#input:
#output: the velocity vector of the spacecraft in the next time step 
#JUST GRAVITY
def update_velocity_justg():
    current_velocity = velocity_vector 
    return current_velocity + a_due_to_gravity()*DT

#with sails on
def update_velocity():
    current_velocity = velocity_vector
    return current_velocity + total_acceleration()*DT

#input:
#output: the position vecotr of the spacecraft in the next time step
def update_position():
    current_position = position_vector
    return current_position + velocity_vector*DT

#input:
#output: returns the correct updated position and velocity vectors 
def update_state_justg():
    next_velocity = update_velocity_justg()
    next_position = update_position()

    return next_position, next_velocity 

#with sails on 
def update_state():
    next_velocity = update_velocity()
    next_position = update_position()

    return next_position, next_velocity 
#########################################################################################


def integrate_justg(time):
    global position_vector, velocity_vector

    t = 0.0

    state_history = [
        np.concatenate((position_vector, velocity_vector, [t]))
    ]

    while t < time:
        next_position, next_velocity = update_state_justg()

        position_vector = next_position
        velocity_vector = next_velocity

        t += DT

        current_state = np.concatenate(
            (position_vector, velocity_vector, [t])
        )

        state_history.append(current_state.copy())

    return np.array(state_history)

def integrate(time):
    global position_vector, velocity_vector

    t = 0.0

    state_history = [
        np.concatenate((position_vector, velocity_vector, [t]))
    ]

    while t < time:
        next_position, next_velocity = update_state()

        position_vector = next_position
        velocity_vector = next_velocity

        t += DT

        current_state = np.concatenate(
            (position_vector, velocity_vector, [t])
        )

        state_history.append(current_state.copy())

    return np.array(state_history)

"""def eq_motion():
    current_state = solar_sail_state
    current_velocity = current_state[3:]

    acceleration = total_acceleration()

    return np.concatenate(current_velocity, acceleration)"""

"""def eq_motion_just_gravity():
    current_state = solar_sail_state
    current_velocity = current_state[3:]

    acceleration = a_due_to_gravity()

    return np.concatenate(current_velocity, acceleration)"""


#integrates upto an orbit, with a max time limit. 
# Returns the history of the integration and whether or not the orbit was reached.
def integrate_to_orbit(orbit_radius, max_time):
    global position_vector, velocity_vector
    t = 0.0 
    
    state_history = [
            np.concatenate((position_vector, velocity_vector, [t]))
        ]

    while np.linalg.norm(position_vector) < orbit_radius and t < max_time:
        next_position, next_velocity = update_state()

        position_vector = next_position
        velocity_vector = next_velocity

        t += DT

        current_state = np.concatenate(
            (position_vector, velocity_vector, [t])
        )

        state_history.append(current_state.copy())

        if np.linalg.norm(position_vector) <= SUN_RADIUS:
            return np.array(state_history), -1000

    reached_orbit = np.linalg.norm(position_vector) >= orbit_radius

    return np.array(state_history), reached_orbit

def integrate_without_history(orbit_radius, max_time):
    global position_vector, velocity_vector

    t = 0.0

    orbit_radius_sq = orbit_radius**2
    sun_radius_sq = SUN_RADIUS**2

    r_sq = np.dot(position_vector, position_vector)

    while r_sq < orbit_radius_sq and t < max_time:

        next_position, next_velocity = update_state()

        position_vector = next_position
        velocity_vector = next_velocity

        t += DT

        r_sq = np.dot(position_vector, position_vector)

        if r_sq <= sun_radius_sq:
            return position_vector.copy(), t, -1000

    reached_orbit = r_sq >= orbit_radius_sq

    return position_vector.copy(), t, reached_orbit

##########################################################
# incoporation of gravitational force from Mars

mars_position_vector = np.array( (MARS_DISTANCE, 0,0) ) #the position vector of mars from the Sun

def sc_to_mars_vector():
    return mars_position_vector - position_vector

def a_due_to_mars():
    distance =  np.linalg.norm(sc_to_mars_vector())
    unit_vector = sc_to_mars_vector()/distance

    magnitude = GRAVITATIONAL * MARS_MASS / (distance ** 2 )
    return magnitude * unit_vector

def total_acceleration_w_mars():
     return a_due_to_gravity() + total_force_radiation() / MASS + a_due_to_mars()

def update_velocity_w_mars():
    current_velocity = velocity_vector
    return current_velocity + DT*total_acceleration_w_mars()

def update_position_w_mars():
    current_position = position_vector
    return current_position + DT*update_velocity_w_mars()

def update_state_w_mars():
    next_velocity = update_velocity_w_mars()
    next_position = update_position_w_mars()

    return next_position, next_velocity

def update_mars_velocity(time, mars_start_phase):
    angular_velocity = math.sqrt( GRAVITATIONAL * SUN_MASS / (MARS_DISTANCE ** 3) )
    magnitude = MARS_DISTANCE * angular_velocity
    x_coord = -math.sin( angular_velocity * time + mars_start_phase )
    y_coord = math.cos( angular_velocity * time + mars_start_phase )

    return magnitude * np.array( [x_coord, y_coord, 0] )

def update_mars_position(time, mars_start_phase):
    angular_velocity = math.sqrt( GRAVITATIONAL * SUN_MASS / (MARS_DISTANCE ** 3) )
    magnitude = MARS_DISTANCE
    x_coord = math.cos(angular_velocity * time + mars_start_phase)
    y_coord = math.sin(angular_velocity * time + mars_start_phase)

    return magnitude * np.array([x_coord, y_coord, 0])

def stage_1(stage_1_distance, max_time, mars_start_phase):
    global position_vector, velocity_vector, mars_position_vector

    mars_position_vector = update_mars_position(0.0, mars_start_phase)

    t = 0.0 
    
    state_history = [
            np.concatenate((position_vector, velocity_vector, [t]))
        ]

    mars_state_history =[
        np.concatenate((mars_position_vector, [t]))
    ]
        

    while np.linalg.norm(sc_to_mars_vector()) > stage_1_distance and t < max_time:
        next_position, next_velocity = update_state_w_mars()

        position_vector = next_position
        velocity_vector = next_velocity

        t += DT

        mars_position_vector = update_mars_position(t, mars_start_phase)

        current_state = np.concatenate(
            (position_vector, velocity_vector, [t])
        )

        mars_current_state = np.concatenate(
            (mars_position_vector, [t])
        )
        

        state_history.append(current_state.copy())

        mars_state_history.append(mars_current_state.copy())

        if np.linalg.norm(position_vector) <= SUN_RADIUS:
            return np.array(state_history), np.array(mars_state_history), "crashed into Sun"
        elif np.linalg.norm(sc_to_mars_vector()) <= MARS_RADIUS:
            return np.array(state_history), np.array(mars_state_history), "crashed into Mars"

    if np.linalg.norm(sc_to_mars_vector()) <= stage_1_distance:
        return np.array(state_history), np.array(mars_state_history), sc_to_mars_vector()
    else:
        return np.array(state_history), np.array(mars_state_history), "Stage 1 fail under time limit"

def stage_1_without_history(stage_1_distance, max_time, mars_start_phase):
    global position_vector, velocity_vector, mars_position_vector

    t = 0.0 

    mars_position_vector = update_mars_position(0.0, mars_start_phase)

    while np.linalg.norm(sc_to_mars_vector()) > stage_1_distance and t < max_time:
        next_position, next_velocity = update_state_w_mars()

        position_vector = next_position
        velocity_vector = next_velocity

        t += DT

        mars_position_vector = update_mars_position(t, mars_start_phase)

        if np.linalg.norm(position_vector) <= SUN_RADIUS:
            return sc_to_mars_vector(), t, "crashed into Sun"
        elif np.linalg.norm(sc_to_mars_vector()) <= MARS_RADIUS:
            return sc_to_mars_vector(), t, "crashed into Mars"

    if np.linalg.norm(sc_to_mars_vector()) <= stage_1_distance:
        return sc_to_mars_vector(), t, "Stage 1 success"
    else:
        return sc_to_mars_vector(), t, "Stage 1 fail under time limit"

#position_vector, velocity_vector = update_state()

# Testing 
#position_vector = np.array([AU, 0.0, 0.0])
#velocity_vector = np.array([0.0, 29780.0, 0.0])

#a_angle = 0
#b_angle = 0

#print(irradiance())
#print(total_force_radiation())