import numpy as np
from filterpy.kalman import KalmanFilter
import utilities as ut

def init_filter(dt):

    states = 18 # number of states, first and second order derivatives
    obs = 6 # observations

    filter = KalmanFilter(dim_x=states,dim_z=obs)

    filter.x = np.zeros([1,states]).T

    transition_mat = np.eye(9)

    for i in range(0,9):

        id_vel = i + 3
        id_acc = i + 6

        if id_vel < 9:
            transition_mat[i,id_vel] = dt
        if id_acc < 9:
            transition_mat[i,id_acc] = 0.5 * dt * dt

    zero_mat = np.zeros([9,9])

    tmp1 = np.hstack([transition_mat,zero_mat])
    tmp2 = np.hstack([zero_mat,transition_mat])

    filter.F = np.vstack([tmp1,tmp2])

    filter.H = np.zeros([obs,states])
    filter.H[0,0] = 1
    filter.H[1,1] = 1
    filter.H[2,2] = 1
    filter.H[3,9] = 1
    filter.H[4,10] = 1
    filter.H[5,11] = 1

    filter.Q = np.eye(states) * 1e-4; # process noise
    filter.R = np.eye(obs) * 0.01 # measurement noise
    filter.P = np.eye(states) * 1e-4 # covariance post
    filter.u = 0.

    return filter


def test_filter(filter):

    for i in range(0,5):

        obs = np.array([[0.1*i,0.05*i,0.02*i,0.01*i,0.005*i,0.002*i]]).T

        filter.predict()

        filter.update(obs)
        print(ut.to_string(obs.T,2))

        res = np.array([filter.x[0,0],filter.x[1,0],filter.x[2,0],filter.x[9,0],filter.x[10,0],filter.x[11,0]])

        print(ut.to_string(filter.x.T,2))
        print(ut.to_string(res.T,2))

    return filter
