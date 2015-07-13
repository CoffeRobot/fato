Pinot Tracker
=====

2D and 3D tracking of unknown objects

# Contents

- [Installation](#markdown-header-installation)
- [Cameras](#markdown-header-cameras)
- [Dependencies](#markdown-header-simulation)
- [Tracker](#markdown-header-pr2)
    - [Tracking 2D](#markdown-header-start-and-stop)
    - [Tracking 3D](#markdown-header-object-tracking)

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
If you installed successfully libfreenect2 on your system, you can add this line in the install file to clone 
this package necessary to use the Kinect V2.
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
- libfreenect2, iai_kinect2: required to use Kinect V2
- libuvc: required to run a general RGB usb camera

## Tracker

### Tracking 2D

Run the following command in a terminal:
```
roslaunch pinot_tracker_nodes usb_camera.launch
```
Once the video is shown on the screen please draw a bounding box aruond the object you want to track by pressing the left mouse button and draggin it.

### Tracking 3D

```
WORK IN PROGRESS...
```

See [here](http://wiki.ros.org/pr2_mechanism_controllers/LaserScannerTrajController) for more details.


