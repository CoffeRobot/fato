Pinot Tracker
=====

2D and 3D tracking of unknown objects

# Contents

- [Installation](#markdown-header-installation)
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
mkdir -p ~/catkin_ws/src
```

Copy the contents of [amazon_challenge.rosinstall](amazon_challenge.rosinstall) into a file ~/catkin_ws/src/.rosinstall
*Note: Bitbucket requires this README to be rendered at a specific commit for the file links to work (e.g. go to source and select the devel branch).*

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


