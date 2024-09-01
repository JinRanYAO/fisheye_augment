# Introduction
This repo proposes a few-shot fisheye images data augmentation method based on historical trajectories. First, the distance between the tractor axle and the hitch when the image is captured is used as the radius of the spherical projection, and then the camera pose is sampled continuously on the existing historical trajectories continuously, and eventually new fisheye images are generated by reprojected with sampled camera pose.

The fisheye images data augmentation method in this code can be used for any scene, not limited to autonomous docking tasks, as long as sample camera motion trajectories are available. The key lies in selecting the projection radius appropriately.

The inputs include real annotated fisheye images(yolo keypoint format) and historical trajectories(.txt), and the outputs include the new generated fisheye images and annotations.
# Environment
Python, Numpy, OpenCV, Scipy
# Example
<p align="center">
    <img src="relative/path/to/ppt1x3.gif" width="300" alt="Example1-vehicle">
    <img src="relative/path/to/another_gif.gif" width="300" alt="Example2-vehicle">
</p>
