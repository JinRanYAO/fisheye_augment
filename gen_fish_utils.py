import cv2
import numpy as np
from numba import jit
from scipy.interpolate import LinearNDInterpolator

def data_to_transform(r_matrix,t_position):
    mat =np.hstack((r_matrix,t_position))
    mat=np.vstack((mat,[0.0,0.0,0.0,1.0]))
    return mat

def create_transformation_matrix(roll, pitch, yaw, x, y, z):
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw),  np.cos(yaw), 0, 0],
        [          0,            0, 1, 0],
        [          0,            0, 0, 1]
    ])

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch), 0],
        [             0, 1,             0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [             0, 0,             0, 1]
    ])

    Rx = np.array([
        [1,           0,            0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll),  np.cos(roll), 0],
        [0,           0,            0, 1]
    ])
    R = Rz.dot(Ry).dot(Rx)
    
    t = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])
    
    transformation_matrix = t.dot(R)
    
    return transformation_matrix

def transform_points(transformation_matrix, points):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed_homogeneous = np.dot(transformation_matrix, points_homogeneous.T)
    points_transformed = points_transformed_homogeneous[:3].T
    return points_transformed

def create_fisheye_mask(image_width, image_height, cx, cy):
    Y, X = np.ogrid[:image_height, :image_width]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center <= 828
    return mask.astype(np.uint8)

def filter_box(box, mask):
    """Filter the box points so that only points inside the valid mask area are kept."""
    filtered_box = [point for point in box if mask[int(point[1]), int(point[0])] > 0]
    return filtered_box

def yolo_to_box_points(detection):
    """Convert YOLO detection format to box pixel points."""
    center_x, center_y, width, height = detection
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)

    box = [(x,y) for x in range(x_min, x_max+1) for y in [y_min, y_max]] + [(x,y) for x in [x_min, x_max] for y in range(y_min+1, y_max)]
    
    return box

@jit(nopython=True)
def filter_and_count(Z, u_distorted, v_distorted, image_with_mask, mapped_image, counts, w, h):
    depth_buffer = np.zeros((h, w))
    for i in range(len(u_distorted)):
        ui, vi = int(u_distorted[i]), int(v_distorted[i])
        if 0 <= ui < w and 0 <= vi < h and Z[i] >= depth_buffer[vi, ui]:
            depth_buffer[vi, ui] = Z[i]
            mapped_image[vi, ui] = image_with_mask[i]
            counts[vi, ui] = 1
            
def new_image(mapped_image, counts, w, h):
    non_zero_counts = counts > 0
    unew, vnew = np.meshgrid(np.arange(w), np.arange(h))
    val_points = np.transpose((vnew[non_zero_counts], unew[non_zero_counts]))
    values = mapped_image[non_zero_counts]
    hole_points = np.transpose((vnew[~non_zero_counts], unew[~non_zero_counts]))
    linear_interpolator = LinearNDInterpolator(val_points, values)
    filled_values = linear_interpolator(hole_points)
    filled_image = mapped_image.copy()
    filled_image[~non_zero_counts] = filled_values

    image_new = np.clip(filled_image, 0, 255).astype(np.uint8)
    return image_new

def compute_sphere_scale(poses, pose_base_num):
    x_base, y_base = poses[pose_base_num, 0], poses[pose_base_num, 1]
    x_unit, y_unit = poses[-1, 0], poses[-1, 1]
    distance = np.sqrt((x_base-x_unit)**2 + (y_base-y_unit)**2)
    return 1 + distance