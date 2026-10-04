import numpy as np
import math

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
MARS_HILL_RADIUS = MARS_DISTANCE * (MARS_MASS / (3*SUN_MASS)) ** (1/3)

SAIL_AREA = 2500 #m^2 #80m^2 approximately using NASA's solar sail from: https://www.nasa.gov/smallspacecraft/what-is-acs3/ 
MASS = 25 #mass of spacecraft kg from: https://ntrs.nasa.gov/api/citations/20230008378/downloads/Wilkie_ACS3_mission_update_ISSS_2023_20230530_rev_a.pdf 
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


def distance_from_sun(position_vector):
    return np.linalg.norm(position_vector)

#input: position vector of solar sail
#outputL acceleration vector from sun on solar sail
# a_g = -GMr/r^3
def a_due_to_gravity(position_vector):
    scalar_term = GRAVITATIONAL * SUN_MASS / ( distance_from_sun(position_vector) ) ** 3 
    return -1*scalar_term * position_vector 

# input: the position vector of the solar sail 
# output: the direction vector of photon reaching solar sail
def sun_unit_vector(position_vector):
    return position_vector / np.linalg.norm(position_vector)


#input: the velocity vector of the solar sail
#output: the vector perpendicular to sun unit vector and perpendicular to the orbital plane 
def N_unit_vector(position_vector, velocity_vector):
    cross_product = np.cross(sun_unit_vector(position_vector), velocity_vector)

    if np.linalg.norm(cross_product) < 1e-12:
        raise ValueError(
            "Orbital plane undefined: velocity is parallel to radial direction."
        )

    return cross_product / np.linalg.norm(cross_product)

#input: the position and velocity vector of the solar sail
#output: the vector perpendicular to sun unit vector in the orbital plane 
def T_unit_vector(position_vector, velocity_vector):
    n = N_unit_vector(position_vector, velocity_vector)
    s = sun_unit_vector(position_vector)
    return np.cross(n,s)

#input: the angle of tilt of the solar sail 
#output: the normal vector of the solar sail
def solar_sail_normal_vector(position_vector, velocity_vector, a_angle, b_angle):
    s = sun_unit_vector(position_vector)
    t = T_unit_vector(position_vector, velocity_vector)
    n = N_unit_vector(position_vector, velocity_vector)

    s_component = math.cos(a_angle)
    t_component = math.sin(a_angle) * math.cos(b_angle)
    n_component = math.sin(a_angle) * math.sin(b_angle)

    return s_component * s + t_component * t + n_component * n 


# input: position of solar sail 
# output: the radiant power per unit area hitting the solar sail at given position (scalar)
def irradiance(position_vector):
    distance = distance_from_sun(position_vector)
    return SUN_IRRADIANCE * (AU/distance)**2

#input: position of solar sail
# output: the momentum imparted on the solar sail per second
def incoming_momentum_ps(position_vector, velocity_vector, a_angle, b_angle):
    current_irradiance = irradiance(position_vector)
    a = max(0, np.dot(sun_unit_vector(position_vector), solar_sail_normal_vector(position_vector, velocity_vector, a_angle, b_angle)))

    #depending on the tilt of the solar sail, flux decreases if tilt is not normal to incoming photons. 
    return current_irradiance * SAIL_AREA * a * sun_unit_vector(position_vector) / SPEED_LIGHT 

#input: position of solar sail, velocity_vector
#output: the force imparted on the solar sail due to specular reflection
def dF_reflection(position_vector, velocity_vector, a_angle, b_angle):
    a = max(0, np.dot(sun_unit_vector(position_vector), solar_sail_normal_vector(position_vector, velocity_vector, a_angle, b_angle)))
    return R * 2 * irradiance(position_vector) * SAIL_AREA / SPEED_LIGHT * a**2 * solar_sail_normal_vector(position_vector, velocity_vector,a_angle, b_angle)

#input: 
#output: the force on the the solar sail due to absorption of photons
def dF_absorbed(position_vector, velocity_vector, a_angle, b_angle):
    return (1-R)*incoming_momentum_ps(position_vector, velocity_vector, a_angle, b_angle)

def a_due_to_mars(position_vector, mars_position_vector):
    sc_to_mars = mars_position_vector - position_vector
    distance =  np.linalg.norm(sc_to_mars)
    unit_vector = sc_to_mars/distance

    magnitude = GRAVITATIONAL * MARS_MASS / (distance ** 2 )
    return magnitude * unit_vector

#input:
#output: the total force on the solar sail due to radiation pressure (vector)
def total_force_radiation(position_vector, velocity_vector, a_angle, b_angle):
    return dF_reflection(position_vector, velocity_vector, a_angle, b_angle) + dF_absorbed(position_vector, velocity_vector, a_angle, b_angle)

def total_acceleration(position_vector, velocity_vector, mars_position_vector, a_angle, b_angle):
    return a_due_to_gravity(
        position_vector
        ) + total_force_radiation(
            position_vector, velocity_vector, a_angle, b_angle
            ) / MASS + a_due_to_mars(
                position_vector, mars_position_vector
            )

def update_state(current_pos, current_vel, mars_position_vector, a_angle, b_angle, dt):
    new_vel = current_vel + total_acceleration(current_pos, current_vel, mars_position_vector, a_angle, b_angle) * dt 
    new_pos = current_pos + new_vel * dt

    return new_pos, new_vel


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



#https://ntrs.nasa.gov/api/citations/19960035775/downloads/19960035775.pdf 
def get_osculating_periapsis(position_sc, velocity_sc, position_mars, velocity_mars):
    r = position_sc - position_mars
    v = velocity_sc - velocity_mars

    mu = GRAVITATIONAL * MARS_MASS
    
    h = np.cross(r, v)

    eccentricity_vec = np.cross(v, h)/mu - r/np.linalg.norm(r)

    e = np.linalg.norm(eccentricity_vec)

    p = np.dot(h,h)/mu
    rp = p/(1 + e)

    rp_altitude = rp - MARS_RADIUS

    return rp_altitude

def make_prediction(position_sc, velocity_sc, current_time, mars_start_phase, a_angle, b_angle, dt=100):
    pos_future = position_sc.copy()
    vel_future = velocity_sc.copy()
    time_future = current_time

    mars_pos = update_mars_position(time_future, mars_start_phase)

    for i in range(5):
        new_pos, new_vel = update_state(pos_future, vel_future, mars_pos, a_angle, b_angle, dt)
        time_future += dt 
        pos_future = new_pos
        vel_future = new_vel
        mars_pos = update_mars_position(time_future, mars_start_phase)

    
    mars_vel = update_mars_velocity(time_future, mars_start_phase)

    r_relative = pos_future - mars_pos
    v_relative = vel_future - mars_vel
                    
    distance_to_mars = np.linalg.norm(r_relative)
                    
    mars_e = (0.5 * np.dot(v_relative, v_relative)) - GRAVITATIONAL * MARS_MASS / distance_to_mars

    return pos_future, vel_future, mars_pos, mars_vel, mars_e

def steering_algorithm(safe_dist, target, current_position, current_velocity, current_time, mars_start_phase, ca, cb, dt):

    ops = []
    candidates = []
    energies = []

    for alpha in ca:
        for beta in cb:
            pos_future, vel_future, mars_pos, mars_vel, mars_e = make_prediction(
                current_position,
                current_velocity,
                current_time,
                mars_start_phase,
                alpha,
                beta,
                dt)
            
            candidates.append((alpha, beta))

            energies.append(mars_e)

            op = get_osculating_periapsis(pos_future, vel_future, mars_pos, mars_vel)
            ops.append(op)

    best_alpha, best_beta = select_candidate(candidates, ops, energies, safe_dist, target)
            
    return best_alpha, best_beta

def select_candidate(candidates, ops, energies, safe_dist, target):
    ops_arr = np.array(ops)
    energies_arr = np.array(energies)
    candidates_arr = np.array(candidates)

    if max(ops) < safe_dist:
        c_index = np.argmax(ops_arr)
        return candidates_arr[c_index]

    target_m = (ops_arr >= safe_dist) & (ops_arr <= target)

    if np.any(target_m):
        valid = np.where(target_m)[0]
        c_index = valid[np.argmin(energies_arr[valid])]
        return candidates_arr[c_index]

    safe_m = ops_arr >= safe_dist

    safe_indices = np.where(safe_m)[0]

    c_index = safe_indices[
        np.argmin(ops_arr[safe_indices])
    ]
    return candidates_arr[c_index]

def get_score(safe_dist, op, target, energy):

    raw = op - target
    energy_penalty = max(energy, 0)/1e7

    if op < safe_dist:
        return np.inf
    elif op < target:
        score =  3*(target - op)/target
    else:
        score =  raw /target
    return score + energy_penalty



def to_mars_integrator(stage_1_distance, max_time, mars_start_phase, start_pos, start_vel, a_angle, b_angle, safe_dist, target, dt1=1000, dt2=100):

    t = 0.0
    history_trajectory = [
        np.concatenate([start_pos, start_vel, [t]])
        ]
    current_pos_vector = start_pos.copy()
    current_vel_vector = start_vel.copy()

    mars_position_vector = update_mars_position(0.0, mars_start_phase)

    while t < max_time:
        next_position, next_velocity = update_state(current_pos_vector, current_vel_vector, mars_position_vector, a_angle, b_angle, dt1)

        current_pos_vector = next_position
        current_vel_vector = next_velocity

        t += dt1

        current_state = np.concatenate(
            [current_pos_vector, current_vel_vector, [t]]
        )

        history_trajectory.append(current_state.copy())

        mars_position_vector = update_mars_position(t, mars_start_phase)

        sc_to_mars_vector = mars_position_vector - current_pos_vector

        if np.linalg.norm(current_pos_vector) <= SUN_RADIUS:
            return history_trajectory, "crashed into Sun"
        
        if np.linalg.norm(sc_to_mars_vector) <= MARS_RADIUS:
            return history_trajectory, "crashed into Mars"

        if np.linalg.norm(sc_to_mars_vector) <= stage_1_distance:
            break

    if np.linalg.norm(sc_to_mars_vector) > stage_1_distance:
        return history_trajectory, "stage 1 fail under time limit"

    print (
    current_pos_vector.copy(),
    current_vel_vector.copy(),
    mars_position_vector.copy(),
    t,
    "stage 1 success"
    )
    
    #... add steering algorithm 
    steering_hist = []
    ca = np.deg2rad(np.arange(0, 91, 5))
    cb = np.deg2rad(np.arange(0, 360, 15))

    control = 5
    step_count = 0

    mars_vel = update_mars_velocity(t, mars_start_phase)

    v_relative = current_vel_vector - mars_vel
    distance_to_mars = np.linalg.norm(current_pos_vector - mars_position_vector)

    mars_e = (0.5 * np.dot(v_relative, v_relative)) - GRAVITATIONAL * MARS_MASS / distance_to_mars
    op_current = get_osculating_periapsis(current_pos_vector, current_vel_vector, mars_position_vector, mars_vel)

    while t < max_time:
        if step_count % control == 0:
            best_alpha, best_beta = steering_algorithm(
                safe_dist,
                target,
                current_pos_vector,
                current_vel_vector,
                t,
                mars_start_phase,
                ca,
                cb,
                dt2
            )

            steering_hist.append([
                best_alpha, best_beta, t
            ])

            print(
            "energy: {:.3e} J/kg, angle: ({:.1f}°, {:.1f}°), "
            "osc. periapsis alt.: {:.1f} km, dist from mars: {:.1f} km, to time limit: {:.4f}".format(
                mars_e,
                np.rad2deg(best_alpha),
                np.rad2deg(best_beta),
                op_current / 1000,
                distance_to_mars/1000,
                t / max_time
            )
        )

        next_pos, next_vel = update_state(
            current_pos_vector,
            current_vel_vector,
            mars_position_vector,
            best_alpha, 
            best_beta,
            dt2
        )
        current_pos_vector = next_pos
        current_vel_vector = next_vel

        t += dt2

        mars_position_vector = update_mars_position(
            t,
            mars_start_phase
        )

        mars_velocity_vector = update_mars_velocity(
            t,
            mars_start_phase
        )

        current_state = np.concatenate([
            current_pos_vector,
            current_vel_vector,
            [t]
        ])

        history_trajectory.append(current_state)

        r_relative = current_pos_vector - mars_position_vector
        v_relative = current_vel_vector - mars_velocity_vector
                
        distance_to_mars = np.linalg.norm(r_relative)
                
        mars_e = (0.5 * np.dot(v_relative, v_relative)) - GRAVITATIONAL * MARS_MASS / distance_to_mars

        op_current = get_osculating_periapsis(current_pos_vector, current_vel_vector, mars_position_vector, mars_velocity_vector)
        
        if np.linalg.norm(current_pos_vector) <= SUN_RADIUS:
            return history_trajectory, steering_hist, "crashed into sun"

        if distance_to_mars <= MARS_RADIUS:
            return history_trajectory, steering_hist, "crashed into mars"

        if mars_e < 0:
            return history_trajectory, steering_hist, "capture"

        step_count += 1

    return history_trajectory, steering_hist, "stage 2 failed under time limit"


def total_acceleration_no_mars(position_vector, velocity_vector, a_angle, b_angle):
    return a_due_to_gravity(
        position_vector
        ) + total_force_radiation(
            position_vector, velocity_vector, a_angle, b_angle
            ) / MASS

def update_state_no_mars(current_pos, current_vel,  a_angle, b_angle, dt):
    new_vel = current_vel + total_acceleration_no_mars(current_pos, current_vel, a_angle, b_angle) * dt 
    new_pos = current_pos + new_vel * dt

    return new_pos, new_vel


def stage_1_modified(start_pos, start_vel, a_angle, b_angle, max_time, dt=1000):
    a_angle = a_angle*math.pi/180
    b_angle = b_angle*math.pi/180
    position_vector = start_pos.copy()
    velocity_vector = start_vel.copy()
    t = 0.0

    while np.linalg.norm(position_vector) < MARS_DISTANCE and t < max_time:

        new_pos, new_vel = update_state_no_mars(
            position_vector,
            velocity_vector,
            a_angle,
            b_angle,
            dt
        )

        position_vector = new_pos
        velocity_vector = new_vel

        t += dt

    if np.linalg.norm(position_vector) < MARS_DISTANCE:
        return np.linalg.norm(position_vector) - MARS_DISTANCE, "did not reach Mars orbital radius"

    theta = np.arctan2(
        position_vector[1],
        position_vector[0]
    )

    mars_speed = np.sqrt(
        GRAVITATIONAL * SUN_MASS / MARS_DISTANCE
    )

    mars_velocity_vector = mars_speed * np.array([
        -np.sin(theta),
         np.cos(theta),
         0.0
    ])

    rel_speed = np.linalg.norm(
        velocity_vector - mars_velocity_vector
    )

    return (a_angle*180/math.pi, b_angle*180/math.pi), rel_speed, t/(3600*24), "successful"