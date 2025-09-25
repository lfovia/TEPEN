import numpy as np
def habitat_camera_intrinsic(config):
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov, 'The configuration of the depth camera should be the same as rgb camera.'
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f,0,xc],
                                 [0,f,zc],
                                 [0,0,1]],np.float32)
    return intrinsic_matrix

def get_pointcloud_from_depth(rgb:np.ndarray,depth:np.ndarray,intrinsic:np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z,filter_x = np.where(depth>-1)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z,filter_x]
    point_values = np.stack([pixel_x,pixel_z,-pixel_y],axis=-1)
    return filter_x,filter_z,point_values,color_values

def translate_to_world(points:np.ndarray,position:np.ndarray,rotation:np.ndarray):
    extrinsic = np.eye(4)
    extrinsic[0:3,0:3] = rotation 
    extrinsic[0:3,3] = position
    world_points = np.matmul(extrinsic,np.concatenate((points,np.ones((points.shape[0],1))),axis=-1).T).T
    return world_points[:,0:3]

# def project_to_camera(points,intrinsic,position,rotation):
#     extrinsic = np.eye(4)
#     extrinsic[0:3,0:3] = rotation
#     extrinsic[0:3,3] = position
#     extrinsic = np.linalg.inv(extrinsic)
#     camera_points = np.concatenate((points,np.ones((points.shape[0]))),axis=-1)
#     camera_points = np.matmul(extrinsic,camera_points.T).T[:,0:3]
#     depth_values = -camera_points[:,2]
#     filter_x = (camera_points[:,0] * intrinsic[0][0] / depth_values + intrinsic[0][2])
#     filter_z = (-camera_points[:,1] * intrinsic[1][1] / depth_values - intrinsic[1][2] + intrinsic[1][2]*2 - 1)
#     return filter_x,filter_z,depth_values

import numpy as np

def project_to_camera(points, intrinsic, position, rotation, image_width, image_height):
    # Ensure points is a 2D array
    points = np.atleast_2d(points)

    # Convert points to homogeneous coordinates
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))  # Add homogeneous coordinate

    # Compute world-to-camera transformation
    rotation_T = rotation.T  # Transpose of rotation matrix
    translation = -rotation_T @ position  # Compute translation in camera frame

    # Construct full 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_T  # Transpose of rotation
    extrinsic[:3, 3] = translation

    # Transform world points to camera coordinates
    camera_points_homogeneous = (extrinsic @ homogeneous_points.T).T  # Nx4 matrix
    camera_points = camera_points_homogeneous[:, :3]  # Extract (x, y, z)

    # Compute depth values (z-coordinates in camera frame)
    depth_values = camera_points[:, 2]

    # Filter out points behind the camera
    valid_mask = depth_values > 0
    if not np.any(valid_mask):
        print("All points are behind the camera!")
        return None, None, None

    camera_points = camera_points[valid_mask]
    depth_values = depth_values[valid_mask]

    # Project onto the image plane using intrinsic matrix
    u = (camera_points[:, 0] * intrinsic[0, 0] / depth_values) + intrinsic[0, 2]
    v = (camera_points[:, 1] * intrinsic[1, 1] / depth_values) + intrinsic[1, 2]

    # Convert to top-left image coordinates
    v = image_height - v  # Flip y-axis

    # Ensure the projected points are within image bounds
    u = np.clip(u, 0, image_width - 1)
    v = np.clip(v, 0, image_height - 1)

    return u, v, depth_values

