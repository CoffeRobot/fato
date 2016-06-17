import numpy as np


def rotation(a,b,c):

    tx = a
    ty = b
    tz = c
    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0],[0,0,1]])

    return np.dot(Rx, np.dot(Ry, Rz))

def get_projection_matrix(v):


    tx = v[0]
    ty = v[1]
    tz = v[2]
    rx = v[3]
    ry = v[4]
    rz = v[5]

    proj = rotation(rx, ry, rz)
    t = np.array([tx,ty,tz])
    t.shape = (3, 1)
    proj = np.append(proj, t, 1)
    proj = np.vstack([proj, np.array([0, 0, 0, 1])])

    return proj


# points = np.array([
#     [1, 6, 0],
#     [2, 5, 2],
#     [3, 7, 4],
#     [4, 10, 3]
# ])
#
# R = rotation(np.deg2rad(90),0,0)
#
# print(R)
