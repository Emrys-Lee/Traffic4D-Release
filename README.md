# Traffic4D
## Traffic4D: Single View Reconstruction of Repetitious Activity Using Longitudinal Self-Supervision
[Project](http://www.cs.cmu.edu/~ILIM/projects/IM/TRAFFIC4D/) | [PDF](http://www.cs.cmu.edu/~ILIM/projects/IM/TRAFFIC4D/pdf/Traffic4D_Longitudinal_iv2021.pdf) | [Poster](http://www.cs.cmu.edu/~ILIM/projects/IM/TRAFFIC4D/images/poster_IV2021.pdf)\
Fangyu Li, N. Dinesh Reddy, Xudong Chen and Srinivasa G. Narasimhan\
Proceedings of IEEE Intelligent Vehicles Symposium (IV'21, Best Paper Award)

## Set up
### Python
Python version: 3.6.9;\
Python packages are in `requirements.txt` for reference.
### C++
Compilier: `clang++-6.0`;\
Libraries:
1. [ceres](http://ceres-solver.org/installation.html)
```
wget https://github.com/ceres-solver/ceres-solver/archive/1.12.0.zip
unzip 1.12.0.zip
cd ceres-solver-1.12.0/
mkdir build
cd build
cmake .. -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
make -j
sudo make install
```
2. [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
```
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip
unzip eigen-3.3.7.zip
sudo mv eigen-3.3.7/ /usr/include/eigen3/
```
3. [pybind](https://github.com/pybind/pybind11)
```
git clone https://github.com/pybind/pybind11
cd pybind11
cmake .
make -j
sudo make install
```
Build Optimization Library:
```
cd Traffic4D-Release/src/ceres
make
```
`ceres_reconstruct.so` and `ceres_spline.so` are generated under `Traffic4D-Release/src/ceres`.

## Dataset
Download dataset and pre-generated results from [here](https:null), and put it under `Traffic4D-Release/`. The directory should be like
```
Traffic4D-Release/
    Data-Traffic4D/
        fifth_morewood/
            fifth_morewood_init.vd
            top_view.png
            images/
                00001.jpg
                00002.jpg
                ...
                06288.jpg
        arterial_kennedy/
            arterial_kennedy_init.vd
            top_view.png
            images/
                <put AI City Challenge frames here>
        ...
```
The input and output paths can be modified in `config/*.yml`.
### Explanation
#### 1. Input Videos
Sample videos in Traffic4D are provided. Note `arterial_kennedy` and `dodge_century` are from [Nvidia AI City Challenge](https://www.aicitychallenge.org/) City-Scale Multi-Camera Vehicle Tracking Challenge Track. Please request the access [here](https://www.aicitychallenge.org/2021-data-access-instructions/). Once get the data, run
```
ffmpeg -i <mtmc-dir>/train/S01/c001/vdo.avi Traffic4D-Release/Data-Traffic4D/arterial_kennedy/images/%05d.jpg
ffmpeg -i <mtmc-dir>/test/S02/c007/vdo.avi Traffic4D-Release/Data-Traffic4D/dodge_century/images/%05d.jpg
```
to decode frames into `images/`.
#### 2. Pre-Generated 2D Results
Detected 2D bounding boxes, keypoints and tracking IDs are stored in `*_init.vd`. Check [Occlusionnet implementation](https://github.com/dineshreddy91/Occlusion_Net) for detecting keypoints.

#### 3. Output folder
The default folder is `Traffic4D-Release/Result/`.

## Experiments
Run to perform longitudinal reconstruction and activity clustering.
```
    cd Traffic4D-Release
    python exp/traffic4d.py config/fifth_morewood.yml reconstruction
    python exp/traffic4d.py config/fifth_morewood.yml clustering
```
### Results
Find these results in the output folder:
1. 2D keypoints: If 3D reconstruction is done, 2D reprojected keypoints will be plotted.
2. 3D reconstructed trajectories and clusters: The clustered 3D trajectories are plotted on the top view map.

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
