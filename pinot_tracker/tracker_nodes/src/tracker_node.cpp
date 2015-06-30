#include <iostream>
#include "input_manager.h"

using namespace std;


int main(int argc, char* argv[])
{

  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "pinot_tracker_node");

  pinot_tracker::InputManager manager;

  ros::shutdown();

	return 0;
}
