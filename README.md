FATO - Fast and Adaptive Tracker of Objects
=====

ROS-package for detecting and tracking unknown objects in 2D and 3D. For more information please refer to the following paper:

*A.Pieropan; N.Bergström; M.Ishikawa; H.Kjellström (2015) [Robust 3D Tracking of Unknown Objects](http://http://www.csc.kth.se/~hedvig/publications/icra_15.pdf). IEEE International Conference on Robotics and Automation.*

# Contents

- [Installation](#markdown-header-installation)
- [Cameras](#markdown-header-cameras)
- [Dependencies](#markdown-header-dependencies)
- [Tracker](#markdown-header-pr2)
    - [Tracking 2D](#markdown-header-traking-2D)
    - [Tracking 3D](#markdown-header-tracking=3D)
    - [Offline Mode](#markdown-header-offline-mode)

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
- git: {local-name: pinot_tracker, uri: 'git@bitbucket.org:robocoffee/pinot_tracker.git', version: master}
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
roslaunch pinot_tracker_nodes usb_camera.launch
```

If you want to use a specific camera please consider to write your own launch file specifing the parameters you desire as 
descibed on the libuvc_camera package.

### Kinect V2

To run Kinect V2 please run the following commands in the terminal:
```
rosrun kinect2_bridge kinect2_bridge
roslaunch pinot_tracker_nodes kinect2.launch
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
roslaunch pinot_tracker_nodes usb_camera.launch
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
roslaunch pinot_tracker_nodes tracker_kinect_v1.launch
```

Or you can run the opencv gui version used for debugging and testing new features:

```
roslaunch pinot_tracker_tests tracker_test_projection.launch
```
### Offline Mode

It is possible to run the tracker offline using a video as input. The parameters needed for the offline mode are included in the configuration file parameters.yaml.
To run the tracker in this modality please write the following command in the terminal:

```
roslaunch pinot_tracker_nodes tracker_offline.launch

```







