#!/usr/bin/env python
# license removed for brevity
import rospy
import scipy
import numpy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import tf

def talker():
    pub = rospy.Publisher('/fato/pose_publisher', Pose, queue_size=10)
    rospy.init_node('move_model', anonymous=True)
    rate = rospy.Rate(30) # 10hz

    quaternion = tf.transformations.quaternion

    pose = Pose()
    pose.position.x = 0
    pose.position.y = 0
    pose.position.z = 0.5
    pose.orientation.



    while not rospy.is_shutdown():

        pub.publish(pose)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
