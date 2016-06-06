#!/usr/bin/env python
# license removed for brevity
import rospy
import scipy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import cv2
import cmath
import transformations as tf

import h5py
from fato_tracker_tests.srv import *

class PoseEstimation:

    render_service = []
    flow_file = []
    prev_points = []
    next_points = []
    focal = 640.507873535
    X = []
    Y = []
    fx = 0
    fy = 0
    cx = 0
    cy = 0
    obj_pos = []
    obj_rot = []
    gt_pos = []
    gt_rot = []
    upd_pose = []

    def __init__(self):
        self.data = []
        self.render_service = rospy.ServiceProxy('render_next', RenderService)
        self.obj_pos = np.array([0.0,0.0,0.5])
        self.obj_rot = np.array([np.pi,0.0,0.0])
        self.upd_pose = self.get_projection_matrix(self.obj_pos, self.obj_rot)

    def read_data(self):

        self.flow_file = h5py.File("/home/alessandro/debug/flow.hdf5", "r")
        self.prev_points = np.array(self.flow_file['prev_points'])
        self.next_points = np.array(self.flow_file['next_points'])
        self.fx = np.array(self.flow_file['fx'])[0]
        self.fy = np.array(self.flow_file['fy'])[0]
        self.cx = np.array(self.flow_file['cx'])[0]
        self.cy = np.array(self.flow_file['cy'])[0]
        self.focal = (self.fx + self.fy) / 2.0

        self.get_matrices()

    def get_matrices(self):

        self.X = np.empty((0,6), float)
        self.Y = np.empty((0,1), float)
        num_pts = self.prev_points.shape[0]

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

                eq_x = np.array([f/mz, 0, -(x/mz), -(x*y)/f, f + (x*x)/f, -y])
                eq_y = np.array([0, f/mz, -(y/mz), -(f + (y*y)/f), x*y/f, x])
                eq_x.shape = (1,6)
                eq_y.shape = (1,6)

                self.X = np.append(self.X, eq_x, axis=0)
                self.X = np.append(self.X, eq_y, axis=0)

                self.Y = np.append(self.Y, xv)
                self.Y = np.append(self.Y, yv)

        print("valid points " + str(valid_pts))

    def move_object(self,tx,ty,tz,a,b,c):

        response = self.render_service(tx+self.obj_pos[0],
                                       ty+self.obj_pos[1],
                                       tz+self.obj_pos[2],
                                       a+self.obj_rot[0],
                                       b+self.obj_rot[1],
                                       c+self.obj_rot[2],
                                       False)

        self.gt_pos = np.array([tx+self.obj_pos[0], ty+self.obj_pos[1],
                                       tz+self.obj_pos[2]])
        self.gt_rot = np.array([a+self.obj_rot[0], b+self.obj_rot[1],
                                       c+self.obj_rot[2]])

        self.Y = np.empty((0,1), float)

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
                                       True)

        self.upd_pose = self.get_projection_matrix(self.obj_pos, self.obj_rot)

    def draw_motion(self):

        im = cv2.imread("/home/alessandro/debug/image.png")

        for i in range(0, self.prev_points.shape[0] ):

            prev_pt = self.prev_points[i]
            next_pt = self.next_points[i]

            cv2.circle(im, (prev_pt[0],prev_pt[1]), 2, np.array([0,255,0]), -1)
            cv2.line(im, (prev_pt[0],prev_pt[1]), (next_pt[0],next_pt[1]), np.array([255,0,0]), 1)

        cv2.imshow("", im)
        cv2.waitKey(0)

    def draw_motion(self, im = None):

        show_image = False

        if im is None:
            im = cv2.imread("/home/alessandro/debug/image.png")
            show_image = True

        for i in range(0, self.prev_points.shape[0] ):

            prev_pt = self.prev_points[i]
            next_pt = self.next_points[i]

            cv2.circle(im, (prev_pt[0],prev_pt[1]), 2, np.array([0,255,0]), -1)
            cv2.line(im, (prev_pt[0],prev_pt[1]), (next_pt[0],next_pt[1]), np.array([255,0,0]), 1)

        if show_image:
            cv2.imshow("", im)
            cv2.waitKey(0)

        return im

    def compute_pose(self):

        return np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.Y)

    def get_projection_matrix(self, t, r):

        proj = tf.rotation(r[0], r[1], r[2])
        t.shape = (3,1)
        proj = np.append(proj, t,1)
        proj = np.vstack([proj, np.array([0,0,0,1])])

        return proj

    def get_pose_axis(self, proj_mat, camera, axis_len):

        axis = np.zeros([4,3])

        center = np.array([0,0,0,1])
        x_axis = np.array([axis_len,0,0,1])
        y_axis = np.array([0,axis_len,0,1])
        z_axis = np.array([0,0,axis_len,1])

        cp = camera.dot(np.dot(proj_mat,center))
        cp = (cp / cp[2]).astype(int)
        axis[0, :] = cp
        xa = camera.dot(np.dot(proj_mat,x_axis))
        xa = (xa / xa[2]).astype(int)
        axis[1, :] = xa
        ya = camera.dot(np.dot(proj_mat,y_axis))
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

        cv2.circle(im, c, 2, np.array([0,255,0]), -1)
        cv2.line(im, c, x, np.array([0,0,255]), 2)
        cv2.line(im, c, y, np.array([0,255,0]), 2)
        cv2.line(im, c, z, np.array([255,0,0]), 2)

        return im

    def draw_projected_transform(self, proj_mat, camera, im):

        axis_len = 0.07

        axis = self.get_pose_axis(proj_mat, camera, axis_len)

        c = (int(axis[0, 0]), int(axis[0, 1]))
        x = (int(axis[1, 0]), int(axis[1, 1]))
        y = (int(axis[2, 0]), int(axis[2, 1]))
        z = (int(axis[3, 0]), int(axis[3, 1]))

        cv2.circle(im, c, 2, np.array([0,255,0]), -1)
        cv2.line(im, c, x, np.array([255,0,255]), 2) # magenta
        cv2.line(im, c, y, np.array([0,255,255]), 2) # yellow
        cv2.line(im, c, z, np.array([255,255,0]), 2) # cyan

        return im

    def project_pose(self):

        im = cv2.imread("/home/alessandro/debug/image.png")
        # 3x4 camera matrix
        cam = np.array([[self.fx,0,self.cx,0],[0,self.fy,self.cy,0],[0,0,1,0]])

        gt_t = self.gt_pos
        gt_r = np.array([self.gt_rot[0],self.gt_rot[1],self.gt_rot[2]])

        proj = self.get_projection_matrix(gt_t, gt_r)

        beta = self.compute_pose()

        est_t = np.array([beta[0],beta[1],beta[2]])
        est_r = np.array([beta[3],-beta[4],beta[5]])

        proj_beta = self.get_projection_matrix(est_t, est_r)

        mat_str = ''
        for i in range(0,4):
            for j in range(0,4):
                mat_str += '{:0.3f}'.format(proj_beta[i,j]) + ' '

            mat_str += '\n'

        print(mat_str)
        self.upd_pose = np.dot(proj_beta, self.upd_pose)

        mat_str = '\n updated pose \n'
        for i in range(0,4):
            for j in range(0,4):
                mat_str += '{:0.3f}'.format(self.upd_pose[i,j]) + ' '

            mat_str += '\n'

        print(mat_str)

        #im = self.draw_transform(proj, cam, im)
        im = self.draw_projected_transform(self.upd_pose, cam, im)
        im = self.draw_motion(im)

        cv2.imshow("", im)
        cv2.waitKey(0)

    def dummy_init(self):

        self.reset_object()
        self.move_object(0,0,0,0,0,0)
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

            self.move_object(acc_x, acc_y, acc_z,0,0,0)
            self.read_data()

            beta = self.compute_pose()
            #m = np.linalg.lstsq(self.X, self.Y)[0]
            res = 'beta '
            #res1 = 'np '
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

        while r_x > 0 or r_y > 0 or r_z > 0 :

            if r_x > 0:
                acc_x += vel
                r_x -= np.rad2deg(vel)
            if r_y > 0:
                acc_y += vel
                r_y -= np.rad2deg(vel)
            if r_z > 0:
                acc_z += vel
                r_z -= np.rad2deg(vel)

            self.move_object(0, 0, 0, acc_x,acc_y,acc_z)
            self.read_data()

            beta = self.compute_pose()
            res = 'beta '
            self.check_position_error()
            for num in beta:
                res += '{:0.3f}'.format(num) + ' '
            print(res)

            self.project_pose()

    def dummy_move(self, t_x, t_y, t_z, r_x, r_y, r_z):

        t_vel =0.002
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
            res = 'beta '
            self.check_position_error()
            for num in beta:
                res += '{:0.3f}'.format(num) + ' '
            print(res)

            self.project_pose()


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
            est_r = np.array([np.pi,0.0,0.0])

            proj_beta = self.get_projection_matrix(est_t, est_r)
            cam = np.array([[self.fx,0,self.cx,0],[0,self.fy,self.cy,0],[0,0,1,0]])

            self.next_points = cam.dot(np.dot(proj_beta,self.prev_points))

    def check_position_error(self):

        num_points = self.prev_points.shape[0]

        Dx = Dy = Dz = Du = Dv = 0

        valid_points = 0

        for i in range(0,num_points):

            prev_pt = self.prev_points[i]
            next_pt = self.next_points[i]

            if prev_pt[2] > 0 and next_pt[2] > 0:

                X_p = (prev_pt[0] - self.cx) * prev_pt[2] / self.fx
                Y_p = (prev_pt[1] - self.cy) * prev_pt[2] / self.fy

                X_n = (next_pt[0] - self.cx) * next_pt[2] / self.fx
                Y_n = (next_pt[1] - self.cy) * next_pt[2] / self.fy

                valid_points +=1

                #mess = 'Dx ' + str(X_p - X_n) + ' Dy ' + str(Y_p - Y_n) + ' Dz ' + str(self.prev_points[i,2] - self.next_points[i,2])
                #print(mess)

                Dx += X_p - X_n
                Dy += Y_p - Y_n
                Dz += prev_pt[2] - next_pt[2]
                Du += prev_pt[0] - next_pt[0]
                Dv += prev_pt[1] - next_pt[1]

        if(valid_points > 0):
            Dx = Dx / float(valid_points)
            Dy = Dy / float(valid_points)
            Dz = Dz / float(valid_points)
            Du = Du / float(valid_points)
            Dv = Dv / float(valid_points)

        mess = 'dx ' + '{:0.3f}'.format(Dx) + ' dy ' + '{:0.3f}'.format(Dy) + ' dz ' \
               + '{:0.3f}'.format(Dz) + ' du ' + '{:0.3f}'.format(Du) \
               + ' dv ' + '{:0.3f}'.format(Dv)

        print(mess)



camera_matrix = np.zeros([3,4])
camera_matrix[0,0] = 1
camera_matrix[1,1] = 1
camera_matrix[2,2] = 1

prev_pts_3d = np.array([[0,0,0.5,1],[3,0,0.5,1],[0,3,0.5,1],[3,3,0.5,1],[2,2,0.5,1]])
next_pts_3d = np.array([[0,0,1,1],[3,0,1,1],[0,3,1,1],[3,3,1,1],[2,2,1,1]])

prev_pts_2d = camera_matrix.dot(prev_pts_3d.transpose())
next_pts_2d = camera_matrix.dot(next_pts_3d.transpose())






#pose = PoseEstimation()
#pose.reset_object()
#pose.move_object(0,0,0,0,0,0)
#pose.read_data()

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
