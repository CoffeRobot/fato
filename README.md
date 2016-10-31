FATO - Fast and Adaptive Tracker of Objects
=====

ROS-package for detecting and tracking objects in 2D and 3D. The package includes different trackers implemented over the years.
The package is open source and you can use that freely even for commercial application. The list of publications explaining the algorithms implemented are the following:

2D Tracking:

1. **Robust and Adaptive Keypoint Based Object Tracking**. Alessandro Pieropan, Niklas Bergström, Hedvig Kjellström and Masatoshi Ishikawa. Advanced Robotics, 2015

3D Tracking of unknown objects:

1. **Robust 3D Tracking of Unknown Objects**. Alessandro Pieropan, Niklas Bergström, Masatoshi Ishikawa and Hedvig Kjellström. IEEE International Conference on Robotics and Automation 2015.
2. **Robust Tracking of Unknown Objects Through Adaptive Size Estimation and Appearance Learning**. Alessandro Pieropan, Niklas Bergström, Masatoshi Ishikawa, Danica Kragic and Hedvig Kjellström. IEEE International Conference on Robotics and Automation 2016.

Model Based tracking:

1. **Real Time Object Pose Estimation and Tracking for GPU Enabled Embedded Systems**. Alessandro Pieropan, Niklas Bergström and Masatoshi Ishikawa. Poster at GPU Technology Conference 2016.

CUDA Akaze implementation:

1. **Feature Descriptors for Tracking by Detection: a Benchmark**. Alessandro Pieropan, Mårten Björkman, Niklas Bergström and Danica Kragic (arXiv:1607.06178).

# Contents

- [Installation](#markdown-header-installation)
- [Cameras](#markdown-header-cameras)
- [Dependencies](#markdown-header-dependencies)
- [Tracker](#markdown-header-tracker)
    - [Tracking 2D](#markdown-header-traking-2D)
    - [Tracking 3D](#markdown-header-tracking=3D)
    - [Offline Mode](#markdown-header-offline-mode)
    - [Model Tracking](#markdown-header-model-tracking)
- [Model Generation](#markdown-header-model-generation)

## Installation

Install wstool:
```
sudo apt-get install python-wstool
```

Create your workspace:
```
mkdir -p ~/[name_of_workspace]/src
```

Copy the contents of the following lines into a file ~/[name_of_workspace]/src/.rosinstall
```
- git: {local-name: fato_tracker, uri: 'https://github.com/CoffeRobot/fato.git', version: master}
```
If you installed successfully [libfreenect2](https://github.com/OpenKinect/libfreenect2) on your system, 
you can add this line in the install file to clone [kinect2 bridge package](https://github.com/code-iai/iai_kinect2) 
necessary to use the Kinect V2.
```
- git: {local-name: iai_kinect2, uri: 'https://github.com/code-iai/iai_kinect2.git', version: master}
```

A copy of the tracker.rosinstall file with all the packages can be found in the install repository folder.

Fetch the code:
```
cd ~/catkin_ws/src
wstool init
wstool update
```

Install the dependencies:
```
cd ~/catkin_ws
sudo rosdep init # only if never run before
rosdep install --from-paths src --ignore-src
```

Build:
```
cd ~/catkin_ws
catkin_make
```

Cmake custom flags:

The tracker defines some global variables used during compilation:

- TRACKER_WITH_GPU: set it to true if you want to compile the tracker using GPUs. 
- CUSTON_OPENCV_PATH: since the gpu tracker uses some function defined in opencv you need to point to you local installation of OpenCV compiled with WITH_GPU enabled. The standard opencv installation does not come with that enabled.
- TRACKER_VERBOSE_LOGGING: flag used for debugging purposes. The tracker prints and save more information during the intermediate steps.

Standalone opencv library:

The version of OpenCV included in ROS does not include the gpu module and the nonfree module. Since they are required by some nodes of the tracker, it is necessary to build opencv from scratch and link it to the project. The instruction to compile the library can be found on the official website. It is quite tricky to link it properly inside ROS. The only working hack I could find is to edit ~/.bashrc file and add the following line:

```
export CMAKE_PREFIX_PATH=path_where_opencv_is_installed:$CMAKE_PREFIX_PATH
```

This prevent ROS to look for OpenCV in its default location. If you have a nicer way to solve this issue please let me know.

## Cameras

The tracker support many cameras, each camera has a different launch file where the input to the tracker can be configured.

### USB Cameras

The general usb camera launch file depends on the libuvc_camera package of ROS. Please check you have in installed on your system before 
using the launch file. You can use a USB camera connected to your computer with the following instruction:
```
roslaunch fato_tracker_nodes usb_camera.launch
```

If you want to use a specific camera please consider to write your own launch file specifing the parameters you desire as 
descibed on the libuvc_camera package.

### Kinect V2

To run Kinect V2 please run the following commands in the terminal:
```
rosrun kinect2_bridge kinect2_bridge
roslaunch fato_tracker_nodes kinect2.launch
```

### Kinect V1

To run Kinect V1 please make sure you have libfreenect installed on your system. Then write the following command to launch the node>
```
roslaunch tracker_cameras kinect_v1.launch
```


## Dependencies

- libfreenect: required to use Kinect V1
- [libfreenect2](https://github.com/OpenKinect/libfreenect2), [iai_kinect2](https://github.com/code-iai/iai_kinect2): required to use Kinect V2
- libuvc,[libuvc_camera](http://wiki.ros.org/libuvc_camera): required to run any common usb camera

## Tracker

### Tracking 2D

Run the following command in a terminal:
```
roslaunch fato_tracker_nodes usb_camera.launch
```
Once the video is shown on the screen please draw a bounding box aruond the object you want to track by pressing the left mouse button and draggin it.

### Tracking 3D

First run the following nodes in a terminal:

```
roslaunch tracker_cameras kinect_v1.launch
```
It launches the kinect v1 with the proper configuration parameters. Before launching the tracker please check that the this node is correctly running. It may happend that the node does not publish the expected topics due to unknown driver problems. 

Now you have to options, you can run the tracker that publishes the results as rostopics using the following comand:

```
roslaunch fato_tracker_nodes tracker_kinect_v1.launch
```

Or you can run the opencv gui version used for debugging and testing new features:

```
roslaunch fato_tracker_tests tracker_test_projection.launch
```
### Offline Mode

It is possible to run the tracker offline using a video as input. The parameters needed for the offline mode are included in the configuration file parameters.yaml.
To run the tracker in this modality please write the following command in the terminal:

```
roslaunch fato_tracker_nodes tracker_offline.launch

```
### Model Tracking

The model-based tracker currently works only with monocular cameras and requires a model.h5. There are a couple of models included in the package but please refer to the model generation section to generate your own model.
The path to the target object h5 should be changed in the launch file.

To run the model based tracker implemented using only the CPU please run:
```
roslaunch fato_tracker_nodes tracker_model.launch
```

To run the model based tracker implemented using CUDA and VisionWorks please run:

```
roslaunch fato_tracker_nodes tracker_model_vx.launch
```

## Model Generation

This tool is a modification of the model generation implemented in [Simtrack](https://github.com/karlpauwels/simtrack.git). There are two big differences, first SIFT is not the exclusive feature descriptor used to generate the model but
it is possible to use other descriptors such as ORB, BRISK, AKAZE. Second the feature descriptors are included in the model only if the descriptor are above a manually defined response. This allows to create smaller models and to make the
detection less noisy. The model generation requires to have a 3D model of the object to track in the .obj format. To run the model generation please run:


```
roslaunch fato_tracker_nodes generate_model.launch

```

Please modify the parameters in the launch file to setup the generation.






