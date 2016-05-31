#!/usr/bin/env python
# license removed for brevity
import rospy
import scipy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import tf
import cv2

import h5py
from fato_tracker_tests.srv import *


pose = PoseEstimation()
pose.move_object()
pose.read_data()
pose.get_matrices()

betaHat = np.linalg.inv(pose.X.T.dot(pose.X)).dot(pose.X.T).dot(pose.Y)

twodecimals = ["%.2f" % v for v in betaHat]

print(twodecimals)

im = cv2.imread("/home/alessandro/debug/image.png")


for i in range(0, pose.prev_points.shape[0] ):

    prev_pt = pose.prev_points[i]
    next_pt = pose.next_points[i]

    cv2.circle(im, (prev_pt[0],prev_pt[1]), 2, np.array([0,255,0]), -1)
    cv2.line(im, (prev_pt[0],prev_pt[1]), (next_pt[0],next_pt[1]), np.array([255,0,0]), 1)

cv2.imshow("", im)
cv2.waitKey(0)