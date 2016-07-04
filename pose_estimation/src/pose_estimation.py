#!/usr/bin/env python
# license removed for brevity
import rospy
import scipy
import numpy as np
import warnings

from std_msgs.msg import String
from geometry_msgs.msg import Pose
import cv2
import cmath
import transformations as tf
import utilities as ut

import h5py
from fato_tracker_tests.srv import *


def get_normal_equations(prev_pts, next_pts, pts_z, camera):

    X = np.empty((0, 6), float)
    Y = np.empty((0, 1), float)

    [r,c] = prev_pts.shape
    if r < c and (r == 3 or r == 4):
        raise AssertionError("Unexpected matrix of points, must be in the form Nx3!")

    num_pts = prev_pts.shape[0]

    fx = camera[0,0]
    fy = camera[1,1]
    cx = camera[0,2]
    cy = camera[1,2]
    focal = (fx + fy)/2.0

    print('fx ' + str(fx) + ' fy ' + str(fy) + ' cx ' + str(cx) + ' cy ' + str(cy) + ' f ' + str(focal))



    for i in range(0, num_pts):

        prev_pt = prev_pts[i]
        next_pt = next_pts[i]
        mz = pts_z[i]

        x = next_pt[0] - cx
        y = next_pt[1] - cy
        xv = next_pt[0] - prev_pt[0]
        yv = next_pt[1] - prev_pt[1]

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
        eq_y[0,3] = -(focal + (y*y) / focal)
        eq_y[0,4] = x * y / focal
        eq_y[0,5] = x

        X = np.append(X, eq_x, axis=0)
        X = np.append(X, eq_y, axis=0)

        Y = np.append(Y, xv)
        Y = np.append(Y, yv)

    return X,Y

def ransac_ls(X,Y,num_iters, min_inliers):

    # pick up right indices
    num_points = X.shape[0] / 2
    num_params = X.shape[1]

    if num_points < 3:
        raise AssertionError("Number of points less than 3, cannot solve system!")

    if num_points < min_inliers:
        raise AssertionError("Unexpected number of inliers, must be < X.shape[0]/2!")

    ids = np.arange(num_points)

    min_residuals = np.finfo(float).max
    best_beta = np.zeros([num_params])
    min_outliers = np.iinfo(int).max

    for i in range(num_iters):

        np.random.shuffle(ids)
        logic_id = np.zeros([2*num_points])
        logic_id[2*ids[0:min_inliers]] = 1
        logic_id[2*ids[0:min_inliers]+1] = 1
        mask= logic_id.nonzero()

        X_ran = X[mask]
        Y_ran = Y[mask]

        beta = least_square(X_ran, Y_ran)
        residuals = abs(np.dot(X,beta) - Y)
        sum_residuals = sum(residuals)

        res_scale = 6.9460 * np.median(residuals)

        #print('median residual in iter ' + str(iter) + ' ' + str(np.median(residuals)))
        if res_scale == 0:
            res_scale = 0.00001

        W = residuals / res_scale
        outliers = sum((W > 1)*1)

        if sum_residuals < min_residuals:
            # print('res ' + str(sum_residuals) + ' beta ' + ut.to_string(beta)) \
            #      + ' outliers ' + str(outliers)
            min_residuals = sum_residuals
            best_beta = beta
            min_outliers = outliers

    return best_beta


def least_square(X,Y):

    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def m_estimator(X,Y, num_iters, beta=None):

    if beta is None:
        beta = least_square(X,Y)

    for iter in range(num_iters):

        residuals = abs(np.dot(X,beta) - Y)
        mess = 'residuals ' + str(residuals.shape)

        res_scale = 6.9460 * np.median(residuals)

        #print('median residual in iter ' + str(iter) + ' ' + str(np.median(residuals)))

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
        # print(X.shape)
        # print(mess)
        # get weighted X'es
        XW = np.tile(W, (1, len(beta))) * X
        mess += ' XW ' + str(XW.shape)
        #print(mess)

        a = np.dot(XW.T, X)
        b = np.dot(XW.T, Y)

        # get the least-squares solution to a linear matrix equation
        beta = np.linalg.lstsq(a,b)[0]
        #print(to_string(beta))

    return beta


class PoseEstimation:
    render_service = []
    flow_file = []
    prev_points = []
    next_points = []
    focal = 640.507873535
    X = []
    Y = []
    beta = []
    W = []
    beta_robuts = []
    outliers = []
    fx = 0
    fy = 0
    cx = 0
    cy = 0
    obj_pos = []
    obj_rot = []
    gt_pos = []
    gt_rot = []
    upd_pose = []
    upd_robust_pose = []
    tr_prev_points = []
    tr_next_points = []
    tr_gtd_points = []
    tr_pnp_pose = []
    tr_flow_pose = []
    result_image = []
    add_flow_noise = False

    def __init__(self):
        self.data = []
        self.render_service = rospy.ServiceProxy('render_next', RenderService)
        self.obj_pos = np.array([0.0, 0.0, 0.5])
        self.obj_rot = np.array([np.pi, 0.0, 0.0])
        self.upd_pose = self.get_projection_matrix(self.obj_pos, self.obj_rot)
        self.upd_robust_pose = self.upd_pose.copy()

    def read_data(self):

        self.flow_file = h5py.File("/home/alessandro/debug/flow.hdf5", "r")
        self.prev_points = np.array(self.flow_file['prev_points'])
        self.next_points = np.array(self.flow_file['next_points'])
        self.fx = np.array(self.flow_file['fx'])[0]
        self.fy = np.array(self.flow_file['fy'])[0]
        self.cx = np.array(self.flow_file['cx'])[0]
        self.cy = np.array(self.flow_file['cy'])[0]
        self.focal = (self.fx + self.fy) / 2.0
        self.tr_prev_points = np.array(self.flow_file['track_prev_pts'])
        self.tr_next_points = np.array(self.flow_file['track_next_pts'])
        self.tr_gtd_points = np.array(self.flow_file['track_depth_gt'])
        pnp_rotation = np.array(self.flow_file['pnp_rot'])
        pnp_rotation.shape = (3, 3)
        pnp_translation = np.array(self.flow_file['pnp_tr'])
        self.tr_pnp_pose = np.hstack([pnp_rotation, pnp_translation])
        self.tr_pnp_pose = np.vstack([self.tr_pnp_pose, np.array([0, 0, 0, 1])])
        self.tr_flow_pose = np.array(self.flow_file['track_flow_pose'])
        self.tr_flow_pose.shape = (4, 4)

        self.create_image()

        if self.add_flow_noise:
            self.add_noise()

        self.get_matrices()

    def get_matrices(self):

        self.X = np.empty((0, 6), float)
        self.Y = np.empty((0, 1), float)

        mask = np.isnan(self.prev_points[:,2]) * 1
        mask = mask != 1

        self.prev_points = self.prev_points[mask,...]
        self.next_points = self.next_points[mask,...]

        num_pts = self.prev_points.shape[0]

        [r,c] = self.prev_points.shape
        if r == 3 or r == 4:
            warnings.warn('points must be in the form Nx3, transposing matrix')

        valid_pts = 0

        for i in range(0, num_pts):

            prev_pt = self.prev_points[i]
            next_pt = self.next_points[i]

            mz = prev_pt[2]
            x = next_pt[0] - self.cx
            y = next_pt[1] - self.cy
            xv = next_pt[0] - prev_pt[0]
            yv = next_pt[1] - prev_pt[1]
            f = self.focal
            # assumption of knowing mz but it has to be > 0
            if not np.isnan(mz) > 0:
                valid_pts += 1

                eq_x = np.array([f / mz, 0, -(x / mz), -(x * y) / f, f + (x * x) / f, -y])
                eq_y = np.array([0, f / mz, -(y / mz), -(f + (y * y) / f), x * y / f, x])
                eq_x.shape = (1, 6)
                eq_y.shape = (1, 6)

                self.X = np.append(self.X, eq_x, axis=0)
                self.X = np.append(self.X, eq_y, axis=0)

                self.Y = np.append(self.Y, xv)
                self.Y = np.append(self.Y, yv)

        print("valid points " + str(valid_pts))

    def move_object(self, tx, ty, tz, a, b, c):

        response = self.render_service(tx + self.obj_pos[0],
                                       ty + self.obj_pos[1],
                                       tz + self.obj_pos[2],
                                       a + self.obj_rot[0],
                                       b + self.obj_rot[1],
                                       c + self.obj_rot[2],
                                       False,
                                       self.add_flow_noise)

        self.gt_pos = np.array([tx + self.obj_pos[0], ty + self.obj_pos[1],
                                tz + self.obj_pos[2]])
        self.gt_rot = np.array([a + self.obj_rot[0], b + self.obj_rot[1],
                                c + self.obj_rot[2]])

        self.Y = np.empty((0, 1), float)

        self.fx = response.fx
        self.fy = response.fy
        self.cx = response.cx
        self.cy = response.cy
        self.focal = (self.fx + self.fy) / 2.0

    def reset_object(self):

        response = self.render_service(self.obj_pos[0],
                                       self.obj_pos[1],
                                       self.obj_pos[2],
                                       self.obj_rot[0],
                                       self.obj_rot[1],
                                       self.obj_rot[2],
                                       True,
                                       self.add_flow_noise)

        self.upd_pose = self.get_projection_matrix(self.obj_pos, self.obj_rot)
        self.upd_robust_pose = self.upd_pose.copy()

    def draw_motion(self, im=None, draw_outliers=False):

        show_image = False

        if im is None:
            im = self.result_image
            show_image = True

        if draw_outliers:
            print("points shape: " + str(self.prev_points.shape[0]) + " outliers " +
                  str(self.outliers.shape))

        for i in range(0, self.prev_points.shape[0]):
            prev_pt = self.prev_points[i]
            next_pt = self.next_points[i]

            color = np.array([0, 255, 0])

            if draw_outliers:
                id = 2*i
                if (self.outliers[id] or self.outliers[id+1]) and draw_outliers:
                    color = np.array([0, 0, 255])
                else:
                    color = np.array([0, 255, 0])

            cv2.circle(im, (prev_pt[0], prev_pt[1]), 2, color, -1)
            cv2.line(im, (prev_pt[0], prev_pt[1]), (next_pt[0], next_pt[1]), np.array([255, 0, 0]), 1)

        if show_image:
            cv2.imshow("", im)
            cv2.waitKey(0)

        return im

    def compute_pose(self):

        return np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.Y)

    def compute_pose_iter(self):

        self.beta_robuts = self.beta.copy()

        for iter in range(10):

            residuals = abs(np.dot(self.X,self.beta_robuts) - self.Y)
            mess = 'residuals ' + str(residuals.shape)

            res_scale = 6.9460 * np.median(residuals)

            if res_scale == 0:
                continue

            self.W = residuals / res_scale
            self.W.shape = (self.W.shape[0],1)
            mess += ' W ' + str(self.W.shape)

            self.outliers = (self.W > 1)*1
            self.W[ self.outliers.nonzero() ] = 0
            mess += ' O ' + str(self.outliers.shape)

            good_values = (self.W != 0)*1

            # calculate robust weights for 'good' points
            # Note that if you supply your own regression weight vector,
            # the final weight is the product of the robust weight and the regression weight.
            tmp = 1 - np.power(self.W[ good_values.nonzero() ], 2)
            self.W[ good_values.nonzero() ] = np.power(tmp, 2)

            mess += ' X ' + str(self.X.shape)
            #print(mess)
            # get weighted X'es
            XW = np.tile(self.W, (1, 6)) * self.X
            mess += ' XW ' + str(XW.shape)
            print(mess)

            a = np.dot(XW.T, self.X)
            b = np.dot(XW.T, self.Y)

            # get the least-squares solution to a linear matrix equation
            self.beta_robuts = np.linalg.lstsq(a,b)[0]

    def get_projection_matrix(self, t, r):

        proj = tf.rotation(r[0], r[1], r[2])
        t.shape = (3, 1)
        proj = np.append(proj, t, 1)
        proj = np.vstack([proj, np.array([0, 0, 0, 1])])

        return proj

    def get_pose_axis(self, proj_mat, camera, axis_len):

        axis = np.zeros([4, 3])

        center = np.array([0, 0, 0, 1])
        x_axis = np.array([axis_len, 0, 0, 1])
        y_axis = np.array([0, axis_len, 0, 1])
        z_axis = np.array([0, 0, axis_len, 1])

        cp = camera.dot(np.dot(proj_mat, center))
        cp = (cp / cp[2]).astype(int)
        axis[0, :] = cp
        xa = camera.dot(np.dot(proj_mat, x_axis))
        xa = (xa / xa[2]).astype(int)
        axis[1, :] = xa
        ya = camera.dot(np.dot(proj_mat, y_axis))
        ya = (ya / ya[2]).astype(int)
        axis[2, :] = ya
        za = camera.dot(np.dot(proj_mat, z_axis))
        za = (za / za[2]).astype(int)
        axis[3, :] = za

        return axis

    def draw_transform(self, proj_mat, camera, im):

        axis_len = 0.07

        axis = self.get_pose_axis(proj_mat, camera, axis_len)

        c = (int(axis[0, 0]), int(axis[0, 1]))
        x = (int(axis[1, 0]), int(axis[1, 1]))
        y = (int(axis[2, 0]), int(axis[2, 1]))
        z = (int(axis[3, 0]), int(axis[3, 1]))

        cv2.circle(im, c, 2, np.array([0, 255, 0]), -1)
        cv2.line(im, c, x, np.array([0, 0, 255]), 2)
        cv2.line(im, c, y, np.array([0, 255, 0]), 2)
        cv2.line(im, c, z, np.array([255, 0, 0]), 2)

        return im

    def draw_projected_transform(self, proj_mat, camera, im):

        axis_len = 0.07

        axis = self.get_pose_axis(proj_mat, camera, axis_len)

        c = (int(axis[0, 0]), int(axis[0, 1]))
        x = (int(axis[1, 0]), int(axis[1, 1]))
        y = (int(axis[2, 0]), int(axis[2, 1]))
        z = (int(axis[3, 0]), int(axis[3, 1]))

        cv2.circle(im, c, 2, np.array([0, 255, 0]), -1)
        cv2.line(im, c, x, np.array([255, 0, 255]), 2)  # magenta
        cv2.line(im, c, y, np.array([0, 255, 255]), 2)  # yellow
        cv2.line(im, c, z, np.array([255, 255, 0]), 2)  # cyan

        return im

    def get_depth_color(self, value):

        vmin = -0.02
        vmax = 0.02

        if value < vmin:
            value = vmin
        elif value > vmax:
            value = vmax

        dv = vmax - vmin
        r = g = b = 1

        if value < (vmin + 0.25 * dv):
            r = 0
            g = 4 * (value - vmin) / dv
        elif value < (vmin + 0.5 * dv):
            r = 0
            b = 1 + 4 * (vmin + 0.25 * dv - value) / dv
        elif value < (vmin + 0.75 * dv):
            r = 4 * (value - vmin - 0.5 * dv) / dv
            b = 0
        else:
            g = 1 + 4 * (vmin + 0.75 * dv - value) / dv
            b = 0

        if np.isnan(r) or np.isnan(g) or np.isnan(b):
            r = b = g = 0

        return (np.array([b, g, r]) * 255).astype(int)

    def draw_difference_gt(self, im):

        num_points = self.tr_prev_points.shape[0]

        for i in range(0, num_points):
            pt = self.tr_next_points[i]

            delta = self.tr_gtd_points[i, 0] - self.tr_prev_points[i, 2]
            c = self.get_depth_color(delta)
            cv2.circle(im, (pt[0], pt[1]), 2, c, -1)

        return im

    def project_results(self):

        flow_im = self.project_pose()

        robust_im = self.project_robust_pose()
        #pnp_im = self.project_pnp()

        tr_im = self.project_track_pose()
        tr_im = self.project_pnp(tr_im)
        tr_im = self.draw_difference_gt(tr_im)

        im = np.hstack([flow_im, np.hstack([robust_im, tr_im])])

        self.print_poses()

        cv2.imshow("", im)
        cv2.waitKey(0)

    def project_track_pose(self):

        im = self.result_image.copy()

        num_points = self.tr_prev_points.shape[0]

        cam = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])

        im = self.draw_projected_transform(self.tr_flow_pose, cam, im)

        for i in range(0, num_points):
            p_pt = self.tr_prev_points[i, 0:2]
            n_pt = self.tr_next_points[i, 0:2]

            cv2.circle(im, (p_pt[0], p_pt[1]), 2, np.array([0, 255, 0]), -1)
            cv2.line(im, (p_pt[0], p_pt[1]), (n_pt[0], n_pt[1]), np.array([255, 0, 0]), 1)

        return im

    def project_pnp(self, im=None):

        pnp_image = []
        if im is None:
            pnp_image = self.result_image.copy()
        else:
            pnp_image = im

        cam = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])

        pnp_image = self.draw_transform(self.tr_pnp_pose, cam, pnp_image)

        return pnp_image

    def project_pose(self):

        im = self.result_image.copy()
        # 3x4 camera matrix
        cam = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])

        gt_t = self.gt_pos
        gt_r = np.array([self.gt_rot[0], 0, -self.gt_rot[1]])

        proj = self.get_projection_matrix(gt_t, gt_r)

        self.beta = self.compute_pose()
        self.compute_pose_iter()

        est_t = np.array([self.beta[0], self.beta[1], self.beta[2]])
        est_r = np.array([self.beta[3], -self.beta[4], self.beta[5]])

        proj_beta = self.get_projection_matrix(est_t, est_r)
        self.upd_pose = np.dot(proj_beta, self.upd_pose)

        est_t = np.array([self.beta_robuts[0], self.beta_robuts[1], self.beta_robuts[2]])
        est_r = np.array([self.beta_robuts[3], -self.beta_robuts[4], self.beta_robuts[5]])

        proj_beta = self.get_projection_matrix(est_t, est_r)
        self.upd_robust_pose = np.dot(proj_beta, self.upd_robust_pose)

        im = self.draw_transform(proj, cam, im)
        im = self.draw_projected_transform(self.upd_pose, cam, im)
        im = self.draw_motion(im)

        return im

    def project_robust_pose(self, im=None):

        res_image = []
        if im is None:
            res_image = self.result_image.copy()
        else:
            res_image = im

        cam = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])

        res_image = self.draw_projected_transform(self.upd_robust_pose, cam, res_image)
        res_image = self.draw_motion(res_image,True)

        return res_image

    def dummy_init(self):

        self.reset_object()
        self.move_object(0, 0, 0, 0, 0, 0)
        self.read_data()

    def dummy_translation(self, t_x, t_y, t_z):

        vel = 0.001

        acc_x = acc_y = acc_z = 0

        while t_x > 0 or t_y > 0 or t_z > 0:

            if t_x > 0:
                acc_x += vel
                t_x -= vel
            if t_y > 0:
                acc_y += vel
                t_y -= vel
            if t_z > 0:
                acc_z += vel
                t_z -= vel

            self.move_object(acc_x, acc_y, acc_z, 0, 0, 0)
            self.read_data()

            beta = self.compute_pose()
            # m = np.linalg.lstsq(self.X, self.Y)[0]
            res = 'beta '
            # res1 = 'np '
            self.check_position_error()
            for num in beta:
                res += '{:0.3f}'.format(num) + ' '

            # for num in m:
            #     res1 += '{:0.3f}'.format(num) + ' '

            print(res)
            #            print(res1)

            self.project_pose()

    def dummy_rotation(self, r_x, r_y, r_z):

        vel = np.deg2rad(1)

        acc_x = acc_y = acc_z = 0

        while r_x > 0 or r_y > 0 or r_z > 0:

            if r_x > 0:
                acc_x += vel
                r_x -= np.rad2deg(vel)
            if r_y > 0:
                acc_y += vel
                r_y -= np.rad2deg(vel)
            if r_z > 0:
                acc_z += vel
                r_z -= np.rad2deg(vel)

            self.move_object(0, 0, 0, acc_x, acc_y, acc_z)
            self.read_data()

            beta = self.compute_pose()
            res = 'beta '
            self.check_position_error()
            for num in beta:
                res += '{:0.3f}'.format(num) + ' '
            print(res)

            self.project_pose()

    def dummy_move(self, t_x, t_y, t_z, r_x, r_y, r_z):

        t_vel = 0.002
        r_vel = np.deg2rad(1)

        tx_acc = ty_acc = tz_acc = rx_acc = ry_acc = rz_acc = 0

        dof_count = 6

        while dof_count > 0:

            dof_count = 0

            if r_x > 0:
                rx_acc += r_vel
                r_x -= np.rad2deg(r_vel)
                dof_count += 1
            if r_y > 0:
                ry_acc += r_vel
                r_y -= np.rad2deg(r_vel)
                dof_count += 1
            if r_z > 0:
                rz_acc += r_vel
                r_z -= np.rad2deg(r_vel)
                dof_count += 1
            if t_x > 0:
                tx_acc += t_vel
                t_x -= t_vel
                dof_count += 1
            if t_y > 0:
                ty_acc += t_vel
                t_y -= t_vel
                dof_count += 1
            if t_z > 0:
                tz_acc += t_vel
                t_z -= t_vel
                dof_count += 1

            self.move_object(tx_acc, ty_acc, tz_acc, rx_acc, ry_acc, rz_acc)
            self.read_data()

            beta = self.compute_pose()
            # res = 'beta '
            self.check_position_error()
            # for num in beta:
            #    res += '{:0.3f}'.format(num) + ' '
            # print(res)

            self.project_results()

    def syntetic_translation(self, t_x, t_y, t_z):

        vel = 0.01

        acc_x = acc_y = acc_z = 0

        while t_x > 0 or t_y > 0 or t_z > 0:

            if t_x > 0:
                acc_x += vel
                t_x -= vel
            if t_y > 0:
                acc_y += vel
                t_y -= vel
            if t_z > 0:
                acc_z += vel
                t_z -= vel

            est_t = np.array([t_x, t_y, t_z])
            est_r = np.array([np.pi, 0.0, 0.0])

            proj_beta = self.get_projection_matrix(est_t, est_r)
            cam = np.array([[self.fx, 0, self.cx, 0], [0, self.fy, self.cy, 0], [0, 0, 1, 0]])

            self.next_points = cam.dot(np.dot(proj_beta, self.prev_points))

    def check_position_error(self):

        num_points = self.prev_points.shape[0]

        Dx = Dy = Dz = Du = Dv = 0

        valid_points = 0

        for i in range(0, num_points):

            prev_pt = self.prev_points[i]
            next_pt = self.next_points[i]

            if prev_pt[2] > 0 and next_pt[2] > 0:
                X_p = (prev_pt[0] - self.cx) * prev_pt[2] / self.fx
                Y_p = (prev_pt[1] - self.cy) * prev_pt[2] / self.fy

                X_n = (next_pt[0] - self.cx) * next_pt[2] / self.fx
                Y_n = (next_pt[1] - self.cy) * next_pt[2] / self.fy

                valid_points += 1

                # mess = 'Dx ' + str(X_p - X_n) + ' Dy ' + str(Y_p - Y_n) + ' Dz ' + str(self.prev_points[i,2] - self.next_points[i,2])
                # print(mess)

                Dx += X_p - X_n
                Dy += Y_p - Y_n
                Dz += prev_pt[2] - next_pt[2]
                Du += prev_pt[0] - next_pt[0]
                Dv += prev_pt[1] - next_pt[1]

        if (valid_points > 0):
            Dx = Dx / float(valid_points)
            Dy = Dy / float(valid_points)
            Dz = Dz / float(valid_points)
            Du = Du / float(valid_points)
            Dv = Dv / float(valid_points)

            # mess = 'dx ' + '{:0.3f}'.format(Dx) + ' dy ' + '{:0.3f}'.format(Dy) + ' dz ' \
            #       + '{:0.3f}'.format(Dz) + ' du ' + '{:0.3f}'.format(Du) \
            #       + ' dv ' + '{:0.3f}'.format(Dv)

            # print(mess)

    def print_poses(self):

        gt_t = self.gt_pos
        gt_r = np.array([self.gt_rot[0], self.gt_rot[2], self.gt_rot[1]])

        proj = self.get_projection_matrix(gt_t, gt_r)

        gt_pose_mess = 'POSES: \n gt pose \n'
        upd_pose_mess = '\n updated pose \n'
        robust_pose_mess = '\n robust pose \n'
        pnp_pose_mess = '\n pnp pose \n'
        flow_pose_mess = '\n flow pose \n'
        for i in range(0, 4):
            for j in range(0, 4):
                gt_pose_mess += '{:0.3f}'.format(proj[i, j]) + ' '
                upd_pose_mess += '{:0.3f}'.format(self.upd_pose[i, j]) + ' '
                pnp_pose_mess += '{:0.3f}'.format(self.tr_pnp_pose[i, j]) + ' '
                flow_pose_mess += '{:0.3f}'.format(self.tr_flow_pose[i, j]) + ' '
                robust_pose_mess += '{:0.3f}'.format(self.upd_robust_pose[i, j]) + ' '
            upd_pose_mess += '\n'
            pnp_pose_mess += '\n'
            flow_pose_mess += '\n'
            gt_pose_mess += '\n'
            robust_pose_mess += '\n'

        print(gt_pose_mess + upd_pose_mess + robust_pose_mess + pnp_pose_mess + flow_pose_mess)

        outliers_count = np.sum(self.outliers)
        print('outliers ' + str(outliers_count))

    def create_image(self):

        self.result_image = cv2.imread("/home/alessandro/debug/image.png")
        # self.result_image = np.hstack([im, np.hstack([im,im])])

    def add_noise(self):

        print("Adding depth noise")

        mu = 0
        sigma = 1

        num_points = self.prev_points.shape[0]
        num_noise = (int)(np.floor(num_points * 0.1))

        ids = np.arange(num_points)
        np.random.shuffle(ids)

        depth_noise = np.random.normal(mu, sigma, num_noise)
        depth_noise *= 0.01

        for i in range(0, num_noise):
            self.prev_points[ids[i],2] = self.prev_points[ids[i],2] + depth_noise[i]


camera_matrix = np.zeros([3, 4])
camera_matrix[0, 0] = 1
camera_matrix[1, 1] = 1
camera_matrix[2, 2] = 1

prev_pts_3d = np.array([[0, 0, 0.5, 1], [3, 0, 0.5, 1], [0, 3, 0.5, 1], [3, 3, 0.5, 1], [2, 2, 0.5, 1]])
next_pts_3d = np.array([[0, 0, 1, 1], [3, 0, 1, 1], [0, 3, 1, 1], [3, 3, 1, 1], [2, 2, 1, 1]])

prev_pts_2d = camera_matrix.dot(prev_pts_3d.transpose())
next_pts_2d = camera_matrix.dot(next_pts_3d.transpose())






# pose = PoseEstimation()
# pose.reset_object()
# pose.move_object(0,0,0,0,0,0)
# pose.read_data()

#
# betaHat =
#
# print(betaHat)
#
# print(betaHat[0])
# print(betaHat[1])
# print(betaHat[2])
# print(betaHat[3])
# print(betaHat[4])
# print(betaHat[5])
