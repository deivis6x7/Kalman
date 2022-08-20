#!/usr/bin/env python3.6

import glob
import os
import sys

# Importing right version of Carla client package
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np

import random
import time
import os

# Ground Truth (GT)
COORDINATES_PARAMS_GEO = ('latitude', 'longitude', 'altitude')
COORDINATES_PARAMS = ('location', 'velocity', 'acceleration', 'angular_velocity')
ROTATION_PARAMS = ('pitch', 'yaw', 'roll')
CONTROL_PARAMS = ('throttle', 'steer', 'brake', 'hand_brake', 'reverse', 'manual_gear_shift', 'gear')
# Global Navigation Satellite System (GNSS)
GNSS_PARAMS = ('latitude', 'longitude', 'altitude')
# Inertial Measurement Unit (IMU)
IMU_COORDINATE_PARAMS = ('accelerometer', 'gyroscope')
IMU_PARAMS = ('compass',)
# Light Detection And Ranging (LiDAR)
LiDAR_PARAMS = ('l',) # !

t = 1/5 # Ajuste experimental no simulador
dt = 1/50 # Sample Rate is 50Hz
dtGNSS = 1/5 # Sample Rate is 5Hz
#dtGNSS = 1/50 # Sample Rate is 50Hz (caso queira simplificar para ajustar com timestamps próximos)

POS_TICK_INTERVAL_GNSS = str(dtGNSS) # in seconds
POS_TICK_INTERVAL = str(dt)  # in seconds

# standard deviations
sGNSS = 0.0
sAccel = 0.0
sRotat = 0.0

# bias
bGNSS = 0.0
#bAccel = 0.0
bRotat = 0.0


def main():
    ticks = []
    actor_list = []
    # File manipulation
    n_output = len([d for d in os.listdir() if d.startswith('out')])
    out_folder = f'out{n_output:02d}'
    os.makedirs(out_folder, exist_ok=True)
    pos_file = open(f'{out_folder}/1_pos.csv', 'w')
    gnssGT_file = open(f'{out_folder}/2_gnssGT.csv', 'w')
    gnss_file = open(f'{out_folder}/3_gnss.csv', 'w')
    imu_file = open(f'{out_folder}/4_imu.csv', 'w')
    #lidar_file = open(f'{out_folder}/5_lidar.csv', 'w')

    client = carla.Client('localhost', 2000) # Client creation.
    client.set_timeout(10.0) # Set client connection timeout in seconds.
    world = client.get_world() # World connection.
    blueprint_library = world.get_blueprint_library() # List of actor blueprints available.

    # https://github.com/carla-simulator/carla/issues/5315
    #settings = world.get_settings()
    #settings.synchronous_mode = True # Enables synchronous mode
    #settings.fixed_delta_seconds = 0.05
    #world.apply_settings(settings)

    def write_pos_labels():
        labels = ['timestamp']
        labels.extend(f'{attr}_{c}' for attr in COORDINATES_PARAMS for c in ('x', 'y', 'z'))
        labels.extend(f'rotation_{attr}' for attr in ROTATION_PARAMS)
        labels.extend(f'control_{attr}' for attr in CONTROL_PARAMS)
        pos_file.write(','.join(labels) + '\n')

    def write_pos_values(w_snapshot):
        coordinates_attrs = (getattr(vehicle, 'get_' + p)() for p in COORDINATES_PARAMS)

        # Get timestamp value
        values = [w_snapshot.platform_timestamp]

        # Get COORDINATES_PARAMS values
        values.extend(getattr(attr, c) for attr in coordinates_attrs for c in ('x', 'y', 'z'))

        # Get ROTATION_PARAMS values
        rotation = vehicle.get_transform().rotation
        values.extend(getattr(rotation, attr) for attr in ROTATION_PARAMS)

        # Get CONTROL_PARAMS values
        control = vehicle.get_control()
        values.extend(getattr(control, attr) for attr in CONTROL_PARAMS)
        pos_file.write(','.join(map(str, values)) + '\n')
    
    def write_gnssGT_labels():
        gnssGT_file.write(','.join(('timestamp',) + GNSS_PARAMS) + '\n')

    def write_gnss_labels():
        gnss_file.write(','.join(('timestamp',) + GNSS_PARAMS) + '\n')

    def write_imu_labels():
        coordinates_attrs = tuple(f'{attr}_{c}' for attr in IMU_COORDINATE_PARAMS for c in ('x', 'y', 'z'))
        imu_file.write(','.join(('timestamp',) + coordinates_attrs + IMU_PARAMS) + '\n')

    def write_lidar_labels():
        coordinates_attrs = tuple(f'{attr}_{c}' for attr in LiDAR_PARAMS for c in ('x', 'y', 'z'))
        lidar_file.write(','.join(('timestamp',) + coordinates_attrs) + '\n')

    def write_gnssGT_values(data):
        snapshot = world.get_snapshot()
        write_pos_values(snapshot)  # Small gambiarra
        values = [snapshot.platform_timestamp]
        values.extend(getattr(data, attr) for attr in GNSS_PARAMS)
        gnssGT_file.write(','.join(map(str, values)) + '\n')       

    def write_gnss_values(data):
        snapshot = world.get_snapshot()
        #write_pos_values(snapshot)  # Small gambiarra - verifique
        values = [snapshot.platform_timestamp]
        values.extend(getattr(data, attr) for attr in GNSS_PARAMS)
        gnss_file.write(','.join(map(str, values)) + '\n')

    def write_imu_values(data):
        values = [world.get_snapshot().platform_timestamp]
        values.extend(getattr(data, attr) for attr in IMU_PARAMS)
        attrs = (getattr(data, attr) for attr in IMU_COORDINATE_PARAMS)
        values.extend(getattr(attr, c) for attr in attrs for c in ('x', 'y', 'z'))
        imu_file.write(','.join(map(str, values)) + '\n')

    def  write_lidar_values(raw):
        values = [world.get_snapshot().platform_timestamp]
        points = np.frombuffer(raw.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        # Isolate the intensity
        intensity = points[:, -1]         
        # Isolate the 3D data
        points = points[:, :-1]

        # We're negating the y to correclty visualize a world that matches
        # what we see in Unreal since Open3D uses a right-handed coordinate system
        points[:, :1] = -points[:, :1]

        #values.extend(points)
        lidar_file.write(','.join(map(str, values)) + '\n')

    def get_autonomous():
        # Get a Tesla vehicle from the blueprint library
        vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.tesla.model3'))
        vehicle_bp.set_attribute('color', '1,11,2') # black piano
        
        # Get a random possible (no conflicts) position to spawn the vehicle and spawn it
        #transform = random.choice(world.get_map().get_spawn_points())
        #vehicle = world.spawn_actor(vehicle_bp, transform)
        spawn_point = world.get_map().get_spawn_points()[64]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Carla in 0.9.7 is throwing SIGABRT after ending when using autopilot
        vehicle.set_autopilot()
        # Register vehicle actor
        actor_list.append(vehicle)
        return vehicle

    def get_camera(vehicle):
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('sensor_tick', '0.3')
        #camera = world.spawn_actor(camera_bp,carla.Transform(),attach_to=vehicle)
        camera = world.spawn_actor(camera_bp,carla.Transform(carla.Location(x=1.5, z=2.4)),attach_to=vehicle)
        actor_list.append(camera)

        return camera

    # ground truth dataset generated with high-accuracy GNSS positional data
    def get_gnssGT(vehicle):
        # Get the GNSS definition
        gnssGT_bp = blueprint_library.find('sensor.other.gnss')
        # Noise seed value - Ground Truth: default
        gnssGT_bp.set_attribute('noise_seed', '0')
        # Altitude, latitude and longitude noise - Ground Truth: default
        gnssGT_bp.set_attribute('noise_alt_stddev', str(sGNSS))
        gnssGT_bp.set_attribute('noise_lat_stddev', str(sGNSS))
        gnssGT_bp.set_attribute('noise_lon_stddev', str(sGNSS))
        # Altitude, latitude and longitude noise - default
        gnssGT_bp.set_attribute('noise_alt_bias', str(bGNSS))
        gnssGT_bp.set_attribute('noise_lat_bias', str(bGNSS))
        gnssGT_bp.set_attribute('noise_lon_bias', str(bGNSS))

        gnssGT_bp.set_attribute('sensor_tick', POS_TICK_INTERVAL) 
        # Spawn the GNSS attached at the center of mass of the vehicle
        gnssGT = world.spawn_actor(gnssGT_bp, carla.Transform(), attach_to=vehicle)
        actor_list.append(gnssGT)
        return gnssGT

    def get_gnss(vehicle):
        # Get the GNSS definition
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        # Noise seed value
        gnss_bp.set_attribute('noise_seed', '10')
        # Altitude, latitude and longitude noise
        #sGNSS = 0.5*8.8*t**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
        #sGNSS = sGNSS**2
        sGNSS = 0.000005
        gnss_bp.set_attribute('noise_alt_stddev', str(sGNSS*10)) # metros
        gnss_bp.set_attribute('noise_lat_stddev', str(sGNSS)) # graus
        gnss_bp.set_attribute('noise_lon_stddev', str(sGNSS)) # graus
        # Altitude, latitude and longitude bias
        #bGNSS = sGNSS*0.1
        bGNSS = sGNSS**2
        gnss_bp.set_attribute('noise_alt_bias', str(bGNSS*10)) # metros
        gnss_bp.set_attribute('noise_lat_bias', str(bGNSS)) # graus
        gnss_bp.set_attribute('noise_lon_bias', str(bGNSS)) # graus

        gnss_bp.set_attribute('sensor_tick', POS_TICK_INTERVAL_GNSS)
        # Spawn the GNSS attached at the center of mass of the vehicle
        gnss = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)
        #gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=0.5, z=0.5)), attach_to=vehicle)
        actor_list.append(gnss)
        return gnss

    def get_imu(vehicle):
        # Get the IMU definition
        imu_bp = blueprint_library.find('sensor.other.imu')
        # Noise seed value
        imu_bp.set_attribute('noise_seed', '10')
        # Accelerometer noise
        #sAccel = 0.05 # assume 0.5m/s2
        #sAccel = sAccel**2
        #sAccel = 0.5
        sAccel = 0.001
        imu_bp.set_attribute('noise_accel_stddev_x', str(sAccel)) # m/s2
        imu_bp.set_attribute('noise_accel_stddev_y', str(sAccel)) # m/s2
        imu_bp.set_attribute('noise_accel_stddev_z', str(sAccel)) # m/s2

        # Gyroscope noise
        #sRotat = 1.0*dt # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
        #sRotat = sRotat**2 
        #sRotat = 0.03
        sRotat = 0.001
        imu_bp.set_attribute('noise_gyro_stddev_x', str(sRotat)) # rad/s
        imu_bp.set_attribute('noise_gyro_stddev_z', str(sRotat)) # rad/s
        imu_bp.set_attribute('noise_gyro_stddev_y', str(sRotat)) # rad/s
        # Gyroscope bias
        #sRotat = sRotat*0.1
        sRotat = sRotat**2
        imu_bp.set_attribute('noise_gyro_bias_x', str(sRotat)) # rad/s
        imu_bp.set_attribute('noise_gyro_bias_y', str(sRotat)) # rad/s
        imu_bp.set_attribute('noise_gyro_bias_z', str(sRotat)) # rad/s

        imu_bp.set_attribute('sensor_tick', POS_TICK_INTERVAL)
        # Spawn the IMU attached at the center of mass of the vehicle
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        #imu = world.spawn_actor(imu_bp, carla.Transform(carla.Location(x=0.5, z=0.5)), attach_to=vehicle)
        actor_list.append(imu)
        return imu

    def get_lidar(vehicle, noise):
        # Get the LiDAR definition
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if(noise):
            lidar_bp.set_attribute('noise_stddev', '0.2') # ajustar depois
        #else:
        #    lidar_bp.set_attribute('dropoff_general_rate', '0.45')
        #    lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
        #    lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
        
        lidar_bp.set_attribute('channels', '16')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '300000')
        lidar_bp.set_attribute('rotation_frequency', '20.0')
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-10.0')

        #lidar_bp.set_attribute('horizontal_fov', '360.0')
        #lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.004')
        #lidar_bp.set_attribute('sensor_tick', POS_TICK_INTERVAL) #verificar
        transform = carla.Transform(carla.Location(x=0, z=1.875))
        lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
        return lidar


    try:
        vehicle = get_autonomous()
        spectator = world.get_spectator()

        def follow(_):
            t = vehicle.get_location()
            t.x -= 5
            #t.y -= 5
            t.z += 5
            spectator.set_location(t)
        ticks.append(world.on_tick(follow))

        print('Initiating writing of position data...')
        write_pos_labels()
        # Write pos labels inside gnss handle function
        #ticks.append(world.on_tick(write_pos_values))   #verifique

        print('Initiating writing of Ground Truth data...') # GNSS
        gnssGT = get_gnssGT(vehicle)
        write_gnssGT_labels()
        gnssGT.listen(write_gnssGT_values)

        print('Initiating writing of GNSS data...')
        gnss = get_gnss(vehicle)
        write_gnss_labels()
        gnss.listen(write_gnss_values)

        print('Initiating writing of IMU data...')
        imu = get_imu(vehicle)
        write_imu_labels()
        imu.listen(write_imu_values)

        print('Initiating camera recording...')
        camera = get_camera(vehicle)
        camera.listen(lambda image: image.save_to_disk(f'{out_folder}/{image.frame:06d}.png'))

        #print('Initiating LiDAR recording...')
        #lidar = get_lidar(vehicle, True) # CARLA 0.9.11 -> noise_seed & noise_LiDAR
        #write_lidar_labels()
        #lidar.listen(lambda data: write_lidar_values(data))

        minutes = 5
        duration = 60*minutes

        print()
        for i in range(duration, 0, -1):
            print(f'Letting car drive for more {i}s...', end='\r')
            time.sleep(1)
        print()
        print('Done!', end='\n\n')
        
    finally:
        print('Destroying ticks...')
        for tick in ticks:
            world.remove_on_tick(tick)

        print('Destroying actors...')
        for actor in actor_list:
            if isinstance(actor, carla.libcarla.Vehicle): # Avoid segfault
                actor.set_autopilot(False)
                continue # Let vehicles for later (segfault)

            actor.destroy()

        time.sleep(0.5)

        # Destroy vehicles
        for actor in actor_list:
            actor.destroy()

        pos_file.close()
        gnss_file.close()
        imu_file.close()
        #lidar_file.close()

        print('done.')


if __name__ == '__main__':
    main()