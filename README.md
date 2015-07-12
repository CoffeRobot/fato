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
- git: {local-name: iai_kinect2, uri: 'https://github.com/code-iai/iai_kinect2.git', version: master}
```

Or copy the content of tracker.rosinstall in the install repository folder.

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

```
roslaunch pr2_gazebo pr2_empty_world.launch
roslaunch pr2_moveit_config move_group.launch
```

## Tracker

### Tracking 2D

Start:
```
ssh user@pr2-c1
robot claim
robot start
```

Activate the Kinect with Yasemin's calibration:
```
cd ~/amazon_challenge_ws/
roslaunch kinect_yasemin/kinect_node.launch
```

Start moveit:
```
cd ~/catkin_ws/
source devel/setup.bash
roslaunch pr2_ft_moveit_config move_group.launch
```

Stop:
```
robot stop
robot release
```

### Tracking 3D

Example service call:
```
rosservice call laser_tilt_controller/set_periodic_cmd '{ command: { header: { stamp: 0 }, profile: "linear" , period: 9 , amplitude: 1 , offset: 0 }}'
```

See [here](http://wiki.ros.org/pr2_mechanism_controllers/LaserScannerTrajController) for more details.


