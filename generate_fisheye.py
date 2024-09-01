import cv2
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
import time

import gen_fish_utils as utils

class FisheyeGenerator():
    def __init__(self):
        self.fx = 561
        self.fy = 561
        self.cx = 945
        self.cy = 550
        self.P = np.array([[561.6960814508801, 0.0, 944.6226743796358],
                            [0.0, 562.8300049353666, 549.5333559154889],
                            [0.0, 0.0, 1.0]], dtype=np.float32)
        self.K = np.array([-0.020368584752305338, 0.005757886906715499, -0.0033249471795029164, 0.00010550159530417938], dtype=np.float32)
        
        self.rot_vec = np.array([1.55714067, 1.6136757, -0.91446259])
        self.trans_vec = np.array([-0.01774497, 0.67593043, -0.3108007])
        
        self.image_width = 1920
        self.image_height = 1080
        
        self._init()

    def _init(self):
        self.rec_m = cv2.Rodrigues(self.rot_vec.reshape(3, 1))[0]
        self.T_cb = utils.data_to_transform(self.rec_m.T,-np.dot(self.rec_m.T, self.trans_vec.reshape(3, 1)))
        self.T_bc = np.linalg.inv(self.T_cb)
        self.mask = utils.create_fisheye_mask(self.image_width, self.image_height, self.cx, self.cy)
        self.flat_mask = self.mask.flatten()
        
        u, v = np.meshgrid(np.arange(self.image_width), np.arange(self.image_height))
        points = np.vstack([u.flatten(), v.flatten()]).transpose().astype(np.float32)
        self.filtered_points = points[self.flat_mask.astype(bool)]
        self.valid_points_num = self.filtered_points.shape[0]

    def load_label(self, label_txt):
        label_raw = np.loadtxt(label_txt)
        detection = (int(self.image_width*label_raw[1]), int(self.image_height*label_raw[2]), int(self.image_width*label_raw[3]), int(self.image_height*label_raw[4]))
        box = utils.yolo_to_box_points(detection)
        filtered_box = utils.filter_box(box, self.mask)
        self.filtered_box = np.array(filtered_box, dtype=np.float32)
        self.valid_box_num = self.filtered_box.shape[0]
        if label_raw[13] != 0:
            self.keypoints = np.array([[label_raw[5]*self.image_width, label_raw[6]*self.image_height],[label_raw[8]*self.image_width, label_raw[9]*self.image_height],[label_raw[11]*self.image_width, label_raw[12]*self.image_height]], dtype=np.float32)
        else:
            self.keypoints = np.array([[label_raw[5]*self.image_width, label_raw[6]*self.image_height],[label_raw[8]*self.image_width, label_raw[9]*self.image_height]], dtype=np.float32)
        
    def unit_sphere_point(self):
        all = np.vstack([self.filtered_points, self.filtered_box, self.keypoints])
        undistorted = cv2.fisheye.undistortPoints(all.reshape(-1, 1, 2), K=self.P, D=self.K, P=self.P)
        normalized_points = (undistorted - (self.P[0, 2], self.P[1, 2])) / (self.P[0, 0], self.P[1, 1])
        x, y = normalized_points[:, :, 0].flatten(), normalized_points[:, :, 1].flatten()
        pho = np.sqrt(x**2 + y**2)
        theta = np.arctan(pho)
        self.unit_X = np.sin(theta) * x / pho
        self.unit_Y = np.sin(theta) * y / pho
        self.unit_Z = np.cos(theta)
        
    def load_base_pose(self, poses, pose_base_num):
        self.poses = poses
        x1, y1, z1 = self.poses[pose_base_num, 0], self.poses[pose_base_num, 1], 0.24
        roll1, pitch1, yaw1 = 0.0, 0.0, self.poses[pose_base_num, 2]
        scale = utils.compute_sphere_scale(self.poses, pose_base_num)
        P1 = np.vstack((self.unit_X * scale, self.unit_Y * scale, self.unit_Z * scale)).T
        T1 = utils.create_transformation_matrix(roll1, pitch1, yaw1, x1, y1, z1)
        P1_base = utils.transform_points(self.T_cb, P1)
        self.unit_P_world = utils.transform_points(T1, P1_base)
        
    def gen_new_image(self, new_pose_num, img_raw):
        x2, y2, z2 = self.poses[new_pose_num, 0], self.poses[new_pose_num, 1], 0.24
        roll2, pitch2, yaw2 = 0.0, 0.0, self.poses[new_pose_num, 2]        
        T2 = utils.create_transformation_matrix(roll2, pitch2, yaw2, x2, y2, z2)
        P2_base = utils.transform_points(np.linalg.inv(T2), self.unit_P_world)
        P2 = utils.transform_points(self.T_bc, P2_base)
        
        X_new, Y_new, Z_new = P2.T

        theta_new = np.arctan2(np.sqrt(X_new**2 + Y_new**2), Z_new)
        theta_distorted = theta_new + self.K[0]*theta_new**3 + self.K[1]*theta_new**5 + self.K[2]*theta_new**7 + self.K[3]*theta_new**9
        u_distorted = self.P[0, 0] * theta_distorted / np.sqrt(X_new**2 + Y_new**2) * X_new + self.P[0, 2]
        v_distorted = self.P[1, 1] * theta_distorted / np.sqrt(X_new**2 + Y_new**2) * Y_new + self.P[1, 2]
        
        iu_distorted, iv_distorted = u_distorted[:self.valid_points_num], v_distorted[:self.valid_points_num]
        self.bu_distorted, self.bv_distorted = u_distorted[self.valid_points_num:self.valid_points_num+self.valid_box_num], v_distorted[self.valid_points_num:self.valid_points_num+self.valid_box_num]
        self.ku_distorted, self.kv_distorted = u_distorted[self.valid_points_num+self.valid_box_num:], v_distorted[self.valid_points_num+self.valid_box_num:]
        
        self.mapped_image = np.zeros((self.image_height, self.image_width, 3))
        self.counts = np.zeros((self.image_height, self.image_width))
        image_flatten = img_raw.reshape(-1, img_raw.shape[2])
        imgae_with_mask = image_flatten[self.flat_mask.astype(bool)]
        utils.filter_and_count(Z_new[:self.valid_points_num], iu_distorted, iv_distorted, imgae_with_mask, self.mapped_image, self.counts, self.image_width, self.image_height)
        self.image_new = utils.new_image(self.mapped_image, self.counts, self.image_width, self.image_height)
        
    def save_new_data(self, img_save_path, label_save_path):
        
        cv2.imwrite(img_save_path, self.image_new)
        
        xmin, xmax = max(np.min(self.bu_distorted), 0), min(np.max(self.bu_distorted), self.image_width)
        ymin, ymax = max(np.min(self.bv_distorted), 0), min(np.max(self.bv_distorted), self.image_height)
        box_x_center = (xmin + xmax) / 2 / self.image_width
        box_y_center = (ymin + ymax) / 2 / self.image_height
        box_x_width = (xmax - xmin) / self.image_width
        box_y_height = (ymax - ymin) / self.image_height
        
        left_x = self.ku_distorted[0] / self.image_width
        left_y = self.kv_distorted[0] / self.image_height
        if 0 <= left_x <= 1 and 0 <= left_y <= 1:
            left_vis = 2.0
        else:
            left_x = 0.0
            left_y = 0.0
            left_vis = 0.0
        right_x = self.ku_distorted[1] / self.image_width
        right_y = self.kv_distorted[1] / self.image_height
        if 0 <= right_x <= 1 and 0 <= right_y <= 1:
            right_vis = 2.0
        else:
            right_x = 0.0
            right_y = 0.0
            right_vis = 0.0
        if len(self.ku_distorted) == 3:
            hitch_x = self.ku_distorted[2] / self.image_width
            hitch_y = self.kv_distorted[2] / self.image_height
            if 0 <= hitch_x <= 1 and 0 <= hitch_y <= 1:
                hitch_vis = 2.0
            else:
                hitch_x = 0.0
                hitch_y = 0.0
                hitch_vis = 0.0
        else:
            hitch_x = 0.0
            hitch_y = 0.0
            hitch_vis = 0.0
        
        with open(label_save_path, 'w') as f:
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    0, box_x_center, box_y_center, box_x_width, box_y_height,
                    left_x, left_y, left_vis, right_x, right_y, right_vis, hitch_x, hitch_y, hitch_vis))
            
def retry_on_failure():
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                time.sleep(1)
        return wrapper
    return decorator 

@retry_on_failure()
def generate_image(data_path, dataset_path, seq, img, total_img_num):
    fisheye_generator = FisheyeGenerator()
    if not os.path.isdir(os.path.join(dataset_path, "images", seq, img.split('.')[0])):
        os.mkdir(os.path.join(dataset_path, "images", seq, img.split('.')[0]))
        os.mkdir(os.path.join(dataset_path, "labels", seq, img.split('.')[0]))
        print("Process "+seq+" seq--- " + img)
    label_txt = os.path.join(data_path, "labels", seq, img.split('.')[0]+'.txt')
    image_raw = cv2.imread(os.path.join(data_path, "images", seq, img))
    image_raw_copy = image_raw.copy()
    fisheye_generator.load_label(label_txt)
    fisheye_generator.unit_sphere_point()
    for trajectory_txt in sorted(os.listdir(os.path.join(data_path, "trajectory"))):
        if not os.path.isdir(os.path.join(dataset_path, "images", seq, img.split('.')[0], trajectory_txt.split('.')[0])):
            os.mkdir(os.path.join(dataset_path, "images", seq, img.split('.')[0], trajectory_txt.split('.')[0]))
            os.mkdir(os.path.join(dataset_path, "labels", seq, img.split('.')[0], trajectory_txt.split('.')[0]))
        trajectory = np.loadtxt(os.path.join(data_path, "trajectory", trajectory_txt))
        pose_num = int(int(img.split('.')[0]) / total_img_num * (trajectory.shape[0]-1))
        fisheye_generator.load_base_pose(trajectory, pose_num)
        for num in tqdm(range(0, trajectory.shape[0], 30)):
            if trajectory[num, 0] == 0 and trajectory[num, 1] == 0:
                continue
            if not os.path.exists(os.path.join(dataset_path, "images", seq, img.split('.')[0], trajectory_txt.split('.')[0], str(num).zfill(4)+'.jpg')):
                fisheye_generator.gen_new_image(num, image_raw_copy)
                img_save_path = os.path.join(dataset_path, "images", seq, img.split('.')[0], trajectory_txt.split('.')[0], str(num).zfill(4)+'.jpg')
                label_save_path = os.path.join(dataset_path, "labels", seq, img.split('.')[0], trajectory_txt.split('.')[0], str(num).zfill(4)+'.txt')
                fisheye_generator.save_new_data(img_save_path, label_save_path)

def main():
    data_path = "/home/tongyao/fisheye_dataset/vir_data_80"
    dataset_path = "/home/tongyao/fisheye_dataset/vir_dataset_80"
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = {}
        for seq in sorted(os.listdir(os.path.join(data_path, "images"))):
            if not os.path.isdir(os.path.join(dataset_path, "images", seq)):
                print("Process "+seq+" seq---")
                os.mkdir(os.path.join(dataset_path, "images", seq))
                os.mkdir(os.path.join(dataset_path, "labels", seq))
            image_files = sorted(os.listdir(os.path.join(data_path, "images", seq)))
            total_img_num = int(image_files[-1].split('.')[0])
            for img in image_files:
                task = executor.submit(generate_image, data_path, dataset_path, seq, img, total_img_num)
                tasks[task] = img
            
        for task in concurrent.futures.as_completed(tasks):
            img = tasks[task]
            try:
                task.result()
            except Exception as exc:
                print(exc)
    
if __name__ == "__main__":
    main()