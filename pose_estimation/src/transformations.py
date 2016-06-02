import numpy as np


def rotation(a,b,c):

    tx = a
    ty = b
    tz = c
    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0],[0,0,1]])

    return np.dot(Rx, np.dot(Ry, Rz))

points = np.array([
    [1, 6, 0],
    [2, 5, 2],
    [3, 7, 4],
    [4, 10, 3]
])

R = rotation(np.deg2rad(90),0,0)

print(R)
