import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_points

import transformations as tf

def to_string(A):

        I = J = 1
        tmp = A.shape
        mess = ''
        if len(tmp) == 1:
            I = tmp[0]
            for i in range(0,I):
                mess += '{:0.3f}'.format(A[i]) + ' '
            mess += '\n'

        else:
            I = tmp[0]
            J = tmp[1]
            for i in range(0,I):
                for j in range(0,J):
                    mess += '{:0.3f}'.format(A[i,j]) + ' '
                mess += '\n'

        return mess


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

def create_points(n):
        xs = randrange(n, -5, 5)
        ys = randrange(n, -5, 5)
        zs = randrange(n, 49, 50)

        points = np.zeros([4,n])
        points[0,:] = xs
        points[1,:] = ys
        points[2,:] = zs
        points[3,:] = 1

        return points

def least_square(X,Y):

    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def m_estimator(X,Y, num_iters):

    beta = least_square(X,Y)

    for iter in range(num_iters):

        residuals = abs(np.dot(X,beta) - Y)
        mess = 'residuals ' + str(residuals.shape)

        res_scale = 6.9460 * np.median(residuals)

        if res_scale == 0:
            break

        W = residuals / res_scale

        W.shape = (W.shape[0],1)
        mess += ' W ' + str(W.shape)

        outliers = (W > 1)*1
        W[ outliers.nonzero() ] = 0
        mess += ' O ' + str(outliers.shape)

        good_values = (W != 0)*1

        # calculate robust weights for 'good' points
        # Note that if you supply your own regression weight vector,
        # the final weight is the product of the robust weight and the regression weight.
        tmp = 1 - np.power(W[ good_values.nonzero() ], 2)
        W[ good_values.nonzero() ] = np.power(tmp, 2)

        mess += ' X ' + str(X.shape)
        #print(mess)
        # get weighted X'es
        XW = np.tile(W, (1, 6)) * X
        mess += ' XW ' + str(XW.shape)
        #print(mess)

        a = np.dot(XW.T, X)
        b = np.dot(XW.T, Y)

        # get the least-squares solution to a linear matrix equation
        beta = np.linalg.lstsq(a,b)[0]

    return beta


class test_data:

    X = []
    Y = []
    B = []
    BP = []
    MB = []
    MBP = []
    fx = 649.6468505859375
    fy = 649.00091552734375
    cx = 322.32084374845363
    cy = 221.2580892472088

    pts = []
    camera = []
    projection = []
    pts_2d = []
    proj_2d =[]
    x_flow = []
    y_flow = []
    d_pts = []


    def get_matrices(self, prev_pts, next_pts, d_pts, fx, fy, cx, cy):

        X = np.empty((0, 6), float)
        Y = np.empty((0, 1), float)

        num_pts = prev_pts.shape[0]

        valid_pts = 0

        focal = (fx + fy)/2.0
        f_x = focal
        f_y = focal

        for i in range(0, num_pts):

            prev_pt = prev_pts[i]
            next_pt = next_pts[i]
            mz = d_pts[i]

            x = next_pt[0] - cx
            y = next_pt[1] - cy
            xv = next_pt[0] - prev_pt[0]
            yv = next_pt[1] - prev_pt[1]

            # assumption of knowing mz but it has to be > 0
            if not np.isnan(mz) > 0:
                valid_pts += 1

                eq_x = np.zeros([1,6])
                eq_y = np.zeros([1,6])

                eq_x[0,0] = focal / mz
                eq_x[0,1] = 0
                eq_x[0,2] = -x/mz
                eq_x[0,3] = - x * y / focal
                eq_x[0,4] = focal + (x * x) / focal
                eq_x[0,5] = -y

                eq_y[0,0] = 0
                eq_y[0,1] = focal / mz
                eq_y[0,2] = -y/mz
                eq_y[0,3] = -focal + (y*y) / focal
                eq_y[0,4] = x * y / focal
                eq_y[0,5] = x

                X = np.append(X, eq_x, axis=0)
                X = np.append(X, eq_y, axis=0)

                Y = np.append(Y, xv)
                Y = np.append(Y, yv)

        return X,Y

    def gen_data(self):

        data = {}
        cam = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])
        self.camera = cam

        n = 10
        pts = create_points(n)
        self.pts = pts

        self.projection = tf.get_projection_matrix(np.array([0.0,0,0,0,0,0]))
        self.proj_pts = np.dot(self.projection, pts)


        pts_2d = cam.dot(pts)
        self.pts_2d = pts_2d / pts_2d[2,:]
        proj_2d = cam.dot(self.proj_pts)
        self.proj_2d = proj_2d / proj_2d[2,:]

        self.x_flow = pts_2d[0,:] - proj_2d[0,:]
        self.y_flow = pts_2d[1,:] - proj_2d[1,:]
        self.d_pts = self.proj_pts[2,:]


    def move_points(self,tx,ty,tz,rx,ry,rz):

        self.projection = tf.get_projection_matrix(np.array([tx,ty,tz,rx,ry,rz]))
        self.proj_pts = np.dot(self.projection, self.pts)

        pts_2d = self.camera.dot(self.pts)
        self.pts_2d = pts_2d / pts_2d[2,:]
        proj_2d = self.camera.dot(self.proj_pts)
        self.proj_2d = proj_2d / proj_2d[2,:]

        self.x_flow = pts_2d[0,:] - proj_2d[0,:]
        self.y_flow = pts_2d[1,:] - proj_2d[1,:]
        self.d_pts = self.proj_pts[2,:]

        [X,Y] = self.get_matrices(pts_2d, proj_2d, self.d_pts, self.fx,self.fy,self.cx,self.cy)

        self.X = X
        self.Y = Y

        return data


    def calculate_ls(self,iters):

        self.B = least_square(self.X,self.Y)
        self.BP = tf.get_projection_matrix(self.B)
        self.MB = m_estimator(self.X,self.Y,iters)
        self.MBP = tf.get_projection_matrix(self.MB)

        return data


    def plot_data(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pts = self.pts
        proj_points = self.proj_pts

        ls_pts = np.dot(self.BP,pts)
        m_pts = np.dot(self.MBP,pts)

        ax.scatter(pts[0,:],pts[1,:],pts[2,:], c='r', marker='o')
        ax.scatter(proj_points[0,:],proj_points[1,:],proj_points[2,:], c='b', marker='^')
        ax.scatter(ls_pts[0,:],ls_pts[1,:],ls_pts[2,:], c='c', marker='x')
        ax.scatter(m_pts[0,:],m_pts[1,:],m_pts[2,:], c='y', marker='.')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        print(to_string(self.projection))
        print(to_string(self.BP))
        print(to_string(self.MBP))

        plt.show()


    def calculate_distance(self):

        pts = self.pts
        proj_pts = self.proj_pts

        ls_pts = np.dot(self.BP,pts)
        m_pts = np.dot(self.MBP,pts)

        print(np.mean(pts - proj_pts,1))
        print(np.mean(ls_pts - proj_pts,1))
        print(np.mean(m_pts - proj_pts,1))

# for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zl, zh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)


data = test_data()

data.gen_data()
data.move_points(1,0,0,0,0,0)
data.calculate_ls(10)
data.calculate_distance()

data.plot_data()

# print(d_pts)
# [fx,fy,cx,cy] = get_camera_params()
# [X,Y] = get_matrices(pts_2d, proj_2d, d_pts, fx,fy,cx,cy)
#
# beta = least_square(X,Y)
# beta_m = m_estimator(X,Y,0)
#
# proj_b = tf.get_projection_matrix(beta)
# proj_mb = tf.get_projection_matrix(beta_m)
# print(to_string(proj_b))
# print(to_string(proj_mb))
#
# ls_pts = np.dot(proj_b,pts)
# m_pts = np.dot(proj_mb,pts)

