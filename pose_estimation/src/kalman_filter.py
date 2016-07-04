import numpy as np
import pylab as pl
#from pykalman import KalmanFilter
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

import filterpy
import cv2

# specify parameters
random_state = np.random.RandomState(0)
transition_matrix = [[1, 0.1], [0, 1]]
transition_offset = [-0.1, 0.1]
observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
observation_offset = [1.0, -1.0]
transition_covariance = np.eye(2)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
initial_state_mean = [5, -5]
initial_state_covariance = [[1, 0.1], [-0.1, 1]]

global mouse_x
global mouse_y

def mouse_callback(event,x,y,flags,param):
    global mouse_x,mouse_y
    mouse_x = x
    mouse_y = y
    print('x ' + str(x) + ' y ' + str(y))

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse_callback)

def init_kalman():

    filter = KalmanFilter(dim_x=4,dim_z=2)

    dt = 0.01

    filter.x = np.array([[0,0,0,0]]).T
    filter.F = np.array([[1,0,dt,0],
                        [0,1,0,dt],
                        [0,0,1,0],
                        [0,0,0,1]])
    #q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    #filter.Q = block_diag(q, q)
    filter.Q = np.eye(4) * 1e-4; # process noise
    print(filter.Q)
    filter.H = np.array([[1, 0, 0,        0],
                      [0,        1, 0, 0]])
    filter.R = np.array([[1, 0],[0, 1]]) # measurement noise
    filter.P = np.eye(4) * 1e-4 # covariance post
    filter.u = 0.

    return filter

mouse_x = 0
mouse_y = 0

filter = init_kalman()

while(1):

    filter.predict()
    filter.update(np.array([[mouse_x], [mouse_y]]))
    # do something with the output
    x = filter.x
    cv2.circle(img, (mouse_x,mouse_y),1,(255,0,0),-1)
    cv2.circle(img, (x[0],x[1]),1,(0,255,0),-1)

    cv2.imshow('image',img)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()

# # sample from model
# kf = KalmanFilter(
#     transition_matrix, observation_matrix, transition_covariance,
#     observation_covariance, transition_offset, observation_offset,
#     initial_state_mean, initial_state_covariance,
#     random_state=random_state
# )
# states, observations = kf.sample(
#     n_timesteps=50,
#     initial_state=initial_state_mean
# )
#
# # estimate state with filtering and smoothing
# filtered_state_estimates = kf.filter(observations)[0]
# smoothed_state_estimates = kf.smooth(observations)[0]
#
# # draw estimates
# pl.figure()
# lines_true = pl.plot(states, color='b')
# lines_filt = pl.plot(filtered_state_estimates, color='r')
# lines_smooth = pl.plot(smoothed_state_estimates, color='g')
# pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
#           ('true', 'filt', 'smooth'),
#           loc='lower right'
# )
# pl.show()


