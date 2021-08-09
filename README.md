# Traffic4D: Single View Reconstruction of Repetitious Activity Using Longitudinal Self-Supervision
[Project](http://www.cs.cmu.edu/~ILIM/projects/IM/TRAFFIC4D/) | [PDF](http://www.cs.cmu.edu/~ILIM/projects/IM/TRAFFIC4D/pdf/Traffic4D_Longitudinal_iv2021.pdf) | [Poster](http://www.cs.cmu.edu/~ILIM/projects/IM/TRAFFIC4D/images/poster_IV2021.pdf)\
Fangyu Li, N. Dinesh Reddy, Xudong Chen and Srinivasa G. Narasimhan\
Proceedings of IEEE Intelligent Vehicles Symposium (IV'21, Best Paper Award)

## Set up
The set up process can be skipped if using docker. Please check "Docker" section.
### Python
Python version 3.6.9 is used. Python packages are in `requirements.txt` .
```
git clone https://github.com/Emrys-Lee/Traffic4D-Release.git
sudo apt-get install python3.6
sudo apt-get install python3-pip
cd Traffic4D-Release
pip3 install -r requirements.txt
```
### C++
Traffic4D uses C++ libraries `ceres` and `pybind` for efficient optimization. `pybind` needs `clang` compiler, so Traffic4D uses `clang` compiler.
#### Install `clang` compiler
```
sudo apt-get install clang++-6.0
```
#### Install prerequisites for `ceres`
```
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse and CXSparse (optional)
sudo apt-get install libsuitesparse-dev
```
#### Download and install `ceres`
```
wget https://github.com/ceres-solver/ceres-solver/archive/1.12.0.zip
unzip 1.12.0.zip
cd ceres-solver-1.12.0/
mkdir build
cd build
cmake ..
make
sudo make install
```
#### Download and install [pybind](https://github.com/pybind/pybind11)
```
git clone https://github.com/pybind/pybind11
cd pybind11
cmake .
make
sudo make install
```
#### Build Traffic4D Optimization Library
```
cd Traffic4D-Release/src/ceres
make
```
`ceres_reconstruct.so` and `ceres_spline.so` are generated under path `Traffic4D-Release/src/ceres/`.

## Dataset
Download dataset and pre-generated results from [here](http://platformpgh.cs.cmu.edu/traffic4d/Data-Traffic4D.zip), and put it under `Traffic4D-Release/`.
```
cd Traffic4D-Release
mv Data-Traffic4D.zip ./
unzip Data-Traffic4D.zip
```
The directory should be like
```
Traffic4D-Release/
    Data-Traffic4D/
    └───fifth_morewood/
        └───fifth_morewood_init.vd
        └───top_view.png
        └───images/
                00001.jpg
                00002.jpg
                ...
                06288.jpg
    └───arterial_kennedy/
        └───arterial_kennedy_init.vd
        └───top_view.png
        └───images/
                <put AI City Challenge frames here>
        ...
```
The input and output paths can be modified in `config/*.yml`.
### Explanation
#### 1. Input Videos
Sample videos in Traffic4D are provided. Note `arterial_kennedy` and `dodge_century` are from [Nvidia AI City Challenge](https://www.aicitychallenge.org/) City-Scale Multi-Camera Vehicle Tracking Challenge Track. Please request the access to the dataset [here](https://www.aicitychallenge.org/2021-data-access-instructions/). Once get the data, run
```
ffmpeg -i <mtmc-dir>/train/S01/c001/vdo.avi Traffic4D-Release/Data-Traffic4D/arterial_kennedy/images/%05d.jpg
ffmpeg -i <mtmc-dir>/test/S02/c007/vdo.avi Traffic4D-Release/Data-Traffic4D/dodge_century/images/%05d.jpg
```
to extract frames into `images/`.
#### 2. Pre-Generated 2D Results
Detected 2D bounding boxes, keypoints and tracking IDs are stored in `*_init.vd`. Check [Occlusionnet implementation](https://github.com/dineshreddy91/Occlusion_Net) for detecting keypoints; [V-IOU](https://github.com/bochinski/iou-tracker) for multi-object tracking.

#### 3. Output folder
Folder `Traffic4D-Release/Result/` will be created by default.

### Intersection Information and Sample Results
<p align="center">
  <img src="/demo/intersection_info_and_sample_results.jpg ">
</p>

## Experiments
Run `python exp/traffic4d.py config/<intersection_name>.yml <action>`. Here YML configuration files for multiple intersections are provided under `config/` folder. `<action>` shoulbe be `reconstruction` or `clustering` to perform longitudinal reconstruction and activity clustering sequentially. For example, below runs Fifth and Morewood intersection.
```
cd Traffic4D-Release
python3 exp/traffic4d.py config/fifth_morewood.yml reconstruction
python3 exp/traffic4d.py config/fifth_morewood.yml clustering
```
### Results
Find these results in the output folder:
1. 2D keypoints: If 3D reconstruction is done, 2D reprojected keypoints will be plotted in `Traffic4D-Release/Result/<intersection_name>_keypoints/`.
2. 3D reconstructed trajectories and clusters: The clustered 3D trajectories are plotted on the top view map as `Traffic4D-Release/Result/<intersection_name>_top_view.jpg`.


## Docker
We provide docker image with dependencies already set up. The steps in "Set up" can be skipped if you use docker image. You still need to download the dataset and put it in under `Traffic4D-Release/`. Then map the git repo into docker container to access the dataset.
```
git clone https://github.com/Emrys-Lee/Traffic4D-Release.git
# pull Traffic4D docker image
docker pull emrysli/traffic4d-release:latest
# create a container. For example, if the cloned repo locates at /home/xxx/Traffic4D-Release, <path to cloned repo> should be "/home/xxx"
# If <path in docker container> is "/home/yyy", then "/home/xxx/Traffic4D-Release" will be mapped as "/home/yyy/Traffic4D-Release" inside the container
docker run -it -v <path to cloned repo>/Traffic4D-Release:<path in docker container>/Traffic4D-Release emrysli/traffic4d-release:latest /bin/bash
# inside container
cd <path in docker container>/Traffic4D-Release/src/ceres
make
cd <path in docker container>/Traffic4D-Release
python3 exp/traffic4d.py config/fifth_morewood.yml reconstruction
python3 exp/traffic4d.py config/fifth_morewood.yml clustering
```

## Trouble Shooting
1. `tkinter` module is missing
```
File "/usr/local/lib/python3.6/dist-packages/matplotlib/backends/_backend_tk.py", line 5, in <module>
    import tkinter as Tk
ModuleNotFoundError: No module named 'tkinter'
```
Solution: Please install `tkinter`.
```
sudo apt-get install python3-tk
```
2. `opencv` import error such as
```
File "/usr/local/lib/python3.6/dist-packages/cv2/__init__.py", line 3, in <module>
    from .cv2 import *
ImportError: libSM.so.6: cannot open shared object file: No such file or directory
```
Solution: please install the missing libraries.
```
sudo apt-get install libsm6  libxrender1 libfontconfig1 libxext6
```
## Citation
### Traffic4D
 ```
@conference{Li-2021-127410,
author = {Fangyu Li and N. Dinesh Reddy and Xudong Chen and Srinivasa G. Narasimhan},
title = {Traffic4D: Single View Reconstruction of Repetitious Activity Using Longitudinal Self-Supervision},
booktitle = {Proceedings of IEEE Intelligent Vehicles Symposium (IV '21)},
year = {2021},
month = {July},
publisher = {IEEE},
keywords = {Self-Supervision, vehicle Detection, 4D Reconstruction, 3D reconstuction, Pose Estimation.},
}
```
### Occlusion-Net
```
@inproceedings{onet_cvpr19,
author = {Reddy, N. Dinesh and Vo, Minh and Narasimhan, Srinivasa G.},
title = {Occlusion-Net: 2D/3D Occluded Keypoint Localization Using Graph Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
pages = {7326--7335},
year = {2019}
}
```
