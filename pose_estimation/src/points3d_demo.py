import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings

import transformations as tf
import pose_estimation as pose_estimate
import utilities as ut

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

def create_image_points(n,w,h):

    xs = randrange(n, 0, w)
    ys = randrange(n, 0, h)
    zs = randrange(n, 0.49, 0.50)

    points = np.zeros([3,n])
    points[0,:] = xs
    points[1,:] = ys
    points[2,:] = 1
    #points[3,:] = 1

    return points, zs

def create_fixed_points(n,cx,cy):

    xs = np.zeros(n*n)
    ys = np.zeros(n*n)
    zs = np.zeros(n*n)

    points = np.ones([3,n*n])

    for i in range(0,n):
        for j in range(0,n):

            val_x = cx + (i - n/2) * 3
            val_y = cy + (j - n/2) * 3

            points[0,i+j*n] = val_x
            points[1,i+j*n] = val_y
            zs = 0.5

    return points,zs



def get_lq_data(prev_pts, next_pts, pts_d, camera):

    X = np.empty((0, 6), float)
    Y = np.empty((0, 1), float)

    [r,c] = prev_pts.shape
    if r < c and (r == 3 or r == 4):
        warnings.warn('points must be in the form Nx3, transposing matrix')
        prev_pts = prev_pts.T
        next_pts = next_pts.T

    num_pts = prev_pts.shape[0]

    valid_pts = 0

    fx = camera[0,0]
    fy = camera[1,1]
    cx = camera[0,2]
    cy = camera[1,2]

    focal = (fx + fy)/2.0
    f_x = focal
    f_y = focal

    for i in range(0, num_pts):

        prev_pt = prev_pts[i]
        next_pt = next_pts[i]
        mz = pts_d[i]

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


def get_camera_matrix():

    fx = 649.6468505859375
    fy = 649.00091552734375
    cx = 322.32084374845363
    cy = 221.2580892472088

    return np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])


class test_data:

    X = []
    Y = []
    X_n = []
    Y_n = []

    ls_beta = []
    ls_pose = []
    me_beta = []
    me_pose = []
    ransac_beta = []
    ransac_pose = []

    ls_noise_beta = []
    ls_noise_pose = []
    me_noise_beta = []
    me_noise_pose = []
    ransac_noise_beta = []
    ransac_noise_pose = []

    fx = 649.6468505859375
    fy = 649.00091552734375
    cx = 322.32084374845363
    cy = 221.2580892472088

    init_pts_2d = []
    init_pts_3d = []

    prev_pts_2d = []
    prev_pts_3d = []
    next_pts_2d = []
    next_pts_3d = []
    x_flow = []
    y_flow = []

    prev_noise_2d = []
    prev_noise_3d = []
    next_noise_2d = []
    next_noise_3d = []
    x_flow_noise = []
    y_flow_noise = []

    gt_position = []
    ls_position = []
    m_position = []
    r_position = []
    ls_position_noise = []
    m_position_noise = []
    r_position_noise = []

    pose_ls = []
    pose_m = []
    pose_gt = []

    camera = []
    projection = []

    d_pts = []

    add_noise = False
    noise_percentage = .2
    noise_mu = 0
    noise_sigma = 1.5

    ls_errors = np.empty((0, 1), float)
    m_errors = np.empty((0, 1), float)
    r_errors = np.empty((0, 1), float)
    ls_noise_errors = np.empty((0, 1), float)
    m_noise_errors = np.empty((0, 1), float)
    r_noise_errors = np.empty((0, 1), float)

    m_iters = 10
    ransac_iters = 5

    def gen_data(self, num_points):

        cam = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])
        self.camera = cam

        camera_inv = np.linalg.inv(cam[0:3,0:3])

        [self.init_pts_2d, pts_d] = create_image_points(num_points, 640, 480)

        self.init_pts_3d = np.dot(camera_inv, self.init_pts_2d[0:3,:]) * pts_d
        self.init_pts_3d = np.vstack([self.init_pts_3d, np.ones(num_points)])

        self.ls_position = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.ls_position_noise = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.m_position = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.m_position_noise = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.r_position = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.r_position_noise = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))

        self.prev_pts_2d = self.init_pts_2d.copy()
        self.prev_pts_3d = self.init_pts_3d.copy()
        self.prev_noise_2d = self.init_pts_2d.copy()
        self.prev_noise_3d = self.init_pts_3d.copy()

    def get_fixed_data(self, num_pts):

        cam = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])
        self.camera = cam

        camera_inv = np.linalg.inv(cam[0:3,0:3])

        [self.init_pts_2d, pts_d] = create_fixed_points(num_pts, self.cx, self.cy)
        num_points = self.init_pts_2d.shape[1]

        self.init_pts_3d = np.dot(camera_inv, self.init_pts_2d[0:3,:]) * pts_d
        self.init_pts_3d = np.vstack([self.init_pts_3d, np.ones(num_points)])

        self.ls_position = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.ls_position_noise = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.m_position = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.m_position_noise = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.r_position = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))
        self.r_position_noise = tf.get_projection_matrix(np.array([0,0,0,0,0,0]))

        self.prev_pts_2d = self.init_pts_2d.copy()
        self.prev_pts_3d = self.init_pts_3d.copy()
        self.prev_noise_2d = self.init_pts_2d.copy()
        self.prev_noise_3d = self.init_pts_3d.copy()

    def move_points(self,tx,ty,tz,rx,ry,rz):

        self.projection = tf.get_projection_matrix(np.array([tx,ty,tz,rx,ry,rz]))
        self.next_pts_3d = np.dot(self.projection, self.prev_pts_3d)
        self.next_pts_2d = self.camera.dot(self.next_pts_3d)
        self.next_pts_2d = self.next_pts_2d / self.next_pts_3d[2,:]
        self.next_noise_2d = self.next_pts_2d.copy()

        print('prev')
        print(ut.to_string(self.prev_pts_2d))
        print('next')
        print(ut.to_string(self.next_pts_2d))

        self.x_flow = self.next_pts_2d[0,:] - self.prev_pts_2d[0,:]
        self.y_flow = self.next_pts_2d[1,:] - self.prev_pts_2d[1,:]
        self.d_pts = self.next_pts_3d[2,:]

        self.x_flow_noise = self.x_flow.copy()
        self.y_flow_noise = self.y_flow.copy()

        num_points = self.next_pts_3d.shape[1]
        ids = np.arange(num_points)
        np.random.shuffle(ids)
        num_elems = int(num_points * self.noise_percentage)
        elems = ids[0:num_elems]

        x_noise = np.random.normal(self.noise_mu, self.noise_sigma, num_elems)
        y_noise = np.random.normal(self.noise_mu, self.noise_sigma, num_elems)

        for i in range(0,num_elems):
            id = elems[i]
            self.x_flow_noise[id] += x_noise[i]
            self.y_flow_noise[id] += y_noise[i]
            self.next_noise_2d[0,id] += x_noise[i]
            self.next_noise_2d[1,id] += y_noise[i]

        #[X,Y] = get_lq_data(self.prev_pts_2d, self.next_pts_2d, self.d_pts, self.camera)
        [X,Y] = pose_estimate.get_normal_equations(self.prev_pts_2d.T, self.next_pts_2d.T, self.d_pts, self.camera)
        [self.X_n, self.Y_n] = pose_estimate.get_normal_equations(self.prev_noise_2d.T, self.next_noise_2d.T, self.d_pts, self.camera)

        self.X = X
        self.Y = Y

        self.calculate_ls(10, 10)

        self.calculate_distance()

        self.prev_pts_2d = self.next_pts_2d.copy()
        self.prev_pts_3d = self.next_pts_3d.copy()

        self.prev_noise_2d = self.next_noise_2d.copy()
        self.prev_noise_3d = self.next_pts_3d.copy()

        #self.plot_data()


    def calculate_ls(self,iters, ransac_iters):

        self.ls_beta = pose_estimate.least_square(self.X, self.Y)
        self.ls_pose = tf.get_projection_matrix(self.ls_beta)
        self.ls_position = self.ls_pose.dot(self.ls_position)

        self.me_beta = pose_estimate.m_estimator(self.X, self.Y, iters)
        self.me_pose = tf.get_projection_matrix(self.me_beta)
        self.m_position = self.me_pose.dot(self.m_position)

        init_beta = pose_estimate.ransac_ls(self.X,self.Y,ransac_iters,3)
        self.ransac_beta = pose_estimate.m_estimator(self.X, self.Y, iters, init_beta)
        self.ransac_pose = tf.get_projection_matrix(self.ransac_beta)
        self.r_position = self.ransac_pose.dot(self.r_position)

        self.ls_noise_beta = pose_estimate.least_square(self.X_n, self.Y_n)
        self.ls_noise_pose = tf.get_projection_matrix(self.ls_noise_beta)
        self.ls_position_noise = self.ls_noise_pose.dot(self.ls_position_noise)

        self.me_noise_beta = pose_estimate.m_estimator(self.X_n, self.Y_n, iters)
        self.me_noise_pose = tf.get_projection_matrix(self.me_noise_beta)
        self.m_position_noise = self.me_noise_pose.dot(self.m_position_noise)

        init_beta = pose_estimate.ransac_ls(self.X,self.Y,50,3)
        self.ransac_noise_beta = pose_estimate.m_estimator(self.X_n, self.Y_n, iters, init_beta)
        self.ransac_noise_pose = tf.get_projection_matrix(self.ransac_noise_beta)
        self.r_position_noise = self.ransac_noise_pose.dot(self.r_position_noise)


    def print_poses(self):

        print('gt pose\n' + ut.to_string(self.projection))
        print('ls pose\n' + ut.to_string(self.ls_pose))
        print('m pose\n' + ut.to_string(self.me_pose))
        print('r pose\n' + ut.to_string(self.ransac_pose))

    def plot_data(self):

        fig = plt.figure()

        l1 = plt.plot(self.ls_errors,color='r',label="lsq")
        l2 =plt.plot(self.m_errors,color='b',label='me')
        l3 =plt.plot(self.r_errors,color='g',label='ran')

        l4 =plt.plot(self.ls_noise_errors,'r--',label='lsq_n')
        l5 =plt.plot(self.m_noise_errors,'b--',label='me_n')
        l6 =plt.plot(self.r_noise_errors,'g--',label='ran_n')

        plt.legend()
        plt.show()

        # ax = fig.add_subplot(111, projection='3d')
        #
        # pts = self.init_pts_3d
        # proj_points = self.next_pts_3d
        #
        # ls_pts = np.dot(self.pose_ls,pts)
        # m_pts = np.dot(self.pose_m,pts)
        #
        # ax.scatter(pts[0,:],pts[1,:],pts[2,:], c='r', marker='o')
        # ax.scatter(proj_points[0,:],proj_points[1,:],proj_points[2,:], c='b', marker='^')
        # ax.scatter(ls_pts[0,:],ls_pts[1,:],ls_pts[2,:], c='c', marker='x')
        # ax.scatter(m_pts[0,:],m_pts[1,:],m_pts[2,:], c='y', marker='.')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        #


        #plt.show()


    def calculate_distance(self):

        pts = self.init_pts_3d
        proj_pts = self.next_pts_3d

        ls_pts = np.dot(self.ls_position, pts)
        m_pts = np.dot(self.m_position, pts)
        r_pts = np.dot(self.r_position, pts)

        ls_pts_noise = np.dot(self.ls_position_noise, pts)
        m_pts_noise = np.dot(self.m_position_noise, pts)
        r_pts_noise = np.dot(self.r_position_noise, pts)

        dist_ls = np.linalg.norm(ls_pts - proj_pts)
        dist_m = np.linalg.norm(m_pts - proj_pts)
        dist_r = np.linalg.norm(r_pts - proj_pts)
        dist_m_noise = np.linalg.norm(m_pts_noise - proj_pts)
        dist_ls_noise = np.linalg.norm(ls_pts_noise - proj_pts)
        dist_r_noise = np.linalg.norm(r_pts_noise - proj_pts)

        self.ls_errors = np.append(self.ls_errors , dist_ls)
        self.m_errors = np.append(self.m_errors, dist_m)
        self.r_errors = np.append(self.r_errors , dist_r)
        self.m_noise_errors = np.append(self.m_noise_errors, dist_m_noise)
        self.ls_noise_errors = np.append(self.ls_noise_errors, dist_ls_noise)
        self.r_noise_errors = np.append(self.r_noise_errors, dist_r_noise)

        #print(np.mean(pts - proj_pts,1))
        #print(np.mean(ls_pts - proj_pts,1))
        #print(np.mean(m_pts - proj_pts,1))

    def add_noise_to_data(self, percentage):

        self.add_noise = True
        self.noise_percentage = percentage


# num_points = 10
# [im1_points, pts_d] = create_image_points(num_points, 640, 480)
#
# im2_points = im1_points.copy()
# im2_points[0,:] = im2_points[0,:] + 5
#
# camera = get_camera_matrix()
# camera_inv = np.linalg.inv(camera[0:3,0:3])
#
# pts1 = np.dot(camera_inv, im1_points[0:3,:]) * pts_d
# pts2 = np.dot(camera_inv, im2_points[0:3,:]) * pts_d
#
# X,Y = get_lq_data(im1_points, im2_points, pts_d, camera)
#
# print(to_string(X))
# print(to_string(Y))
#
# beta = least_square(X,Y)
# beta_m = m_estimator(X,Y,10)
#
# print(to_string(beta))
# print(to_string(beta_m))
#
# print(to_string(pts2-pts1))
#
# print('projection 3d rather than 2d')
# projection = tf.get_projection_matrix(np.array([0.00384,0,0,0,0,0]))
# pts1 = np.vstack([pts1, np.ones(num_points)])
# pts3 = np.dot(projection, pts1)
#
# print("pts3")
# print(to_string(pts3))
# print("pts1 - pts3")
# print(to_string(pts1 - pts3))
# im3_points = camera.dot(pts3) / pts3[2,:]
# print("im3pts")
# print(to_string(im3_points))
# print("im1pts")
# print(to_string(im1_points))
# print("im1pts - im3pts")
# print(to_string(im1_points - im3_points))
#
# X,Y = get_lq_data(im1_points, im3_points, pts_d, camera)
#
# beta = least_square(X,Y)
# beta_m = m_estimator(X,Y,10)
#
# print(to_string(beta))
# print(to_string(beta_m))


# for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zl, zh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)


# data = test_data()
#
# data.gen_data()
# data.move_points(1,0,0,0,0,0)
# data.calculate_ls(10)
# data.calculate_distance()
#
# data.plot_data()

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


def plot_errors(ls_err, m_err, r_err, lsn_err, mn_err, rn_err):

    fig = plt.figure()
    plt.subplot(131)
    l1 = plt.plot(ls_err[0,:],color='r',label="lsq")
    l2 =plt.plot(m_err[0,:],color='b',label='me')
    l3 =plt.plot(r_err[0,:],color='g',label='ran')

    l4 =plt.plot(lsn_err[0,:],'r--',label='lsq_n')
    l5 =plt.plot(mn_err[0,:],'b--',label='me_n')
    l6 =plt.plot(rn_err[0,:],'g--',label='ran_n')

    plt.subplot(132)
    l1 = plt.plot(ls_err[1,:],color='r',label="lsq")
    l2 =plt.plot(m_err[1,:],color='b',label='me')
    l3 =plt.plot(r_err[1,:],color='g',label='ran')

    l4 =plt.plot(lsn_err[1,:],'r--',label='lsq_n')
    l5 =plt.plot(mn_err[1,:],'b--',label='me_n')
    l6 =plt.plot(rn_err[1,:],'g--',label='ran_n')

    plt.subplot(133)
    l1 = plt.plot(ls_err[2,:],color='r',label="lsq")
    l2 =plt.plot(m_err[2,:],color='b',label='me')
    l3 =plt.plot(r_err[2,:],color='g',label='ran')

    l4 =plt.plot(lsn_err[2,:],'r--',label='lsq_n')
    l5 =plt.plot(mn_err[2,:],'b--',label='me_n')
    l6 =plt.plot(rn_err[2,:],'g--',label='ran_n')

    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.subplot(131)
    l1 = plt.plot(ls_err[3,:],color='r',label="lsq")
    l2 =plt.plot(m_err[3,:],color='b',label='me')
    l3 =plt.plot(r_err[3,:],color='g',label='ran')

    l4 =plt.plot(lsn_err[3,:],'r--',label='lsq_n')
    l5 =plt.plot(mn_err[3,:],'b--',label='me_n')
    l6 =plt.plot(rn_err[3,:],'g--',label='ran_n')

    plt.subplot(132)
    l1 = plt.plot(ls_err[4,:],color='r',label="lsq")
    l2 =plt.plot(m_err[4,:],color='b',label='me')
    l3 =plt.plot(r_err[4,:],color='g',label='ran')

    l4 =plt.plot(lsn_err[4,:],'r--',label='lsq_n')
    l5 =plt.plot(mn_err[4,:],'b--',label='me_n')
    l6 =plt.plot(rn_err[4,:],'g--',label='ran_n')

    plt.subplot(133)
    l1 = plt.plot(ls_err[5,:],color='r',label="lsq")
    l2 =plt.plot(m_err[5,:],color='b',label='me')
    l3 =plt.plot(r_err[5,:],color='g',label='ran')

    l4 =plt.plot(lsn_err[5,:],'r--',label='lsq_n')
    l5 =plt.plot(mn_err[5,:],'b--',label='me_n')
    l6 =plt.plot(rn_err[5,:],'g--',label='ran_n')

    plt.legend()
    plt.show()

def run_esperiment():

    data = test_data()
    data.fx = 600
    data.fy = 600
    data.cx = 320
    data.cy = 240
    #data.gen_data(30)
    data.get_fixed_data(5)

    angle = np.deg2rad(2)
    gt_tr = np.array([0.0,0.01,0.01,0,angle,0])
    num_iterations = 10

    ls_errors = np.zeros([6,10])
    m_errors = np.zeros([6,10])
    r_errors = np.zeros([6,10])

    lsn_errors = np.zeros([6,10])
    mn_errors = np.zeros([6,10])
    rn_errors = np.zeros([6,10])

    for i in range(0,num_iterations):

        data.move_points(gt_tr[0],gt_tr[1],gt_tr[2],gt_tr[3],gt_tr[4],gt_tr[5])

        ls_errors[:,i] = abs(gt_tr - data.ls_beta)
        m_errors[:,i] = abs(gt_tr - data.me_beta)
        r_errors[:,i] = abs(gt_tr - data.ransac_beta)

        lsn_errors[:,i] = abs(gt_tr - data.ls_noise_beta)
        mn_errors[:,i] = abs(gt_tr - data.me_noise_beta)
        rn_errors[:,i] = abs(gt_tr - data.ransac_noise_beta)

        precision = 5
        print('gt   ' + ut.to_string(gt_tr))
        print('lsq  ' + ut.to_string(data.ls_beta,precision))
        print('m    ' + ut.to_string(data.me_beta,precision))
        print('r    ' + ut.to_string(data.ransac_beta,precision))
        print('lssn ' + ut.to_string(data.ls_noise_beta,precision))
        print('mn   ' + ut.to_string(data.me_noise_beta,precision))
        print('rn   ' + ut.to_string(data.ransac_noise_beta,precision))
        print('\n')



        #data.print_poses()

    plot_errors(ls_errors, m_errors, r_errors, lsn_errors, mn_errors, rn_errors)

    return data

def run_esperiment_dummy():

    # coefficients of the model
    a1, a2, a3 = 0.1, -0.2, 4.0

    # ground truth
    A_gt = [a1, a2, a3]

    #print 'A_gt = ', A_gt

    # create a coordinate matrix
    nx = np.linspace(-1, 1, 41)
    ny = np.linspace(-1, 1, 41)
    x, y = np.meshgrid(nx, ny)

    # make the estimation
    z = a1*x + a2*y + a3

    # let's add some gaussian noise
    z_noise = z + 0.1*np.random.standard_normal(z.shape)

    x_fl = x.flatten()
    y_fl = y.flatten()
    z_ones = np.ones([x.size,1])

    X = np.hstack((np.reshape(x_fl, ([len(x_fl),1])), np.reshape(y_fl, ([len(y_fl),1])), z_ones))

    Z = np.zeros(z_noise.shape)
    Z[:] = z_noise
    Z_fl = Z.flatten()
    Z = np.reshape(Z_fl, ([len(Z_fl),1]))

    # create outliers
    outlier_prop = 0.3
    outlier_IND = np.random.permutation(x.size)
    outlier_IND = outlier_IND[0:np.floor(x.size * outlier_prop)]

    z_noise_outlier = np.zeros(z_noise.shape)
    z_noise_outlier[:] = z_noise
    z_noise_outlier_flt = z_noise_outlier.flatten()

    z_noise_outlier_flt[outlier_IND] = z_noise_outlier_flt[outlier_IND] + 10*np.random.standard_normal(z_noise_outlier_flt[outlier_IND].shape)
    z_noise_outlier = np.reshape(z_noise_outlier_flt, z.shape)

    # non-robust least squares estimation
    Z = np.zeros(z_noise_outlier.shape)
    Z[:] = z_noise_outlier
    Z_fl = Z.flatten()
    Z = np.reshape(Z_fl, ([len(Z_fl),1]))

    beta = pose_estimate.least_square(X,Z)
    beta_m = pose_estimate.m_estimator(X,Z,5)
    beta_r = pose_estimate.ransac_ls(X,Z,10,5)
    beta_r = pose_estimate.m_estimator(X,Z,5,beta_r)

    z_lsq_outlier = np.dot(X, beta)
    z_lsq_outlier = np.reshape(z_lsq_outlier, z.shape)

    z_m_outlier = np.dot(X, beta_m)
    z_m_outlier = np.reshape(z_m_outlier, z.shape)

    z_r_outlier = np.dot(X, beta_r)
    z_r_outlier = np.reshape(z_r_outlier, z.shape)

    lsq_non_robust_outlier = np.hstack((z, z_noise_outlier, z_lsq_outlier, z_m_outlier, z_r_outlier))

    plt.figure()
    plt.title('Non-robust estimate (corrupted by noise AND outliers)')
    plt.imshow(lsq_non_robust_outlier)
    plt.clim(z.min(), z.max())

    plt.show()

#run_esperiment()

# ransac_ls(data.X,data.Y,5,3)

#

data = test_data()
data.get_fixed_data(2)
data.move_points(0.00,0,0,0,np.deg2rad(1),0)

#run_esperiment_dummy()