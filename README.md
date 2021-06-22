# structure_from_motion

The Aim is to reconstruct a 3D structure, given a sequence of images.

# Methodology
  1. first step is to compute the trjectory from sequence of images using Visual Odometry pipeline.
  2. pointcloud is then triangulated from two images. Here calculated the 3d points are the detected feature point from the pair of images which were used to estimate the trajectory. 
  3. the pointcloud is the associated and can be visualized in meshlab.
# Run
```bash
    git clone https://github.com/sushlokshah/structure_from_motion.git
    cd structure_from_motion
      #Edit info.yaml file
      #dataset_location: [image_sequence_path]
      #k: [calibration matrix]
    python3 sfm.py
      #Result is obtained in output.ply at folder specified, which can be viewed using MeshLab, Open3D, or any other related software or library to view point clouds.
```
# Result
|Reconstructed structure| Trajectory | 
| -------- | -------- |
| ![](https://github.com/sushlokshah/structure_from_motion/blob/main/3d_reconstruction.gif) | ![](https://github.com/sushlokshah/structure_from_motion/blob/main/3d_reconstruction_trajectory.gif) |
