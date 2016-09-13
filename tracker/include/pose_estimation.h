/*****************************************************************************/
/*  Copyright (c) 2015, Alessandro Pieropan                                  */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <Eigen/Dense>
#include <utility>

#include "glm/glm.hpp"
#include "../../utilities/include/constants.h"

namespace fato {

void getPoseRansac(const std::vector<cv::Point3f>& model_points,
                   const std::vector<cv::Point2f>& tracked_points,
                   const cv::Mat& camera_model, int iterations, float distance,
                   std::vector<int>& inliers, cv::Mat& rotation,
                   cv::Mat& translation);

void getMatrices(const std::vector<cv::Point2f>& prev_pts,
                 const std::vector<float>& prev_depth,
                 const std::vector<cv::Point2f>& next_pts, float nodal_x,
                 float nodal_y, float focal_x, float focal_y,
                 Eigen::MatrixXf& A, Eigen::VectorXf& b);

/**
 * @brief getPoseFromFlow: estimate the pose of the target from optical flow
 * @param prev_pts:  2d points in the previous frame
 * @param prev_depth: estimated depth in the previous frame
 * @param next_pts: 2d points position estimated by optical flow
 * @param nodal_x: nodal x of the camera
 * @param nodal_y: notal y of the camera
 * @param focal_x
 * @param focal_y
 * @param translation: estimated tx,ty,tz
 * @param rotation: estimated wx,wy,wz
 */
void getPoseFromFlow(const std::vector<cv::Point2f>& prev_pts,
                     const std::vector<float>& prev_depth,
                     const std::vector<cv::Point2f>& next_pts, float nodal_x,
                     float nodal_y, float focal_x, float focal_y,
                     std::vector<float>& translation,
                     std::vector<float>& rotation);

/**
 * @brief getPoseFromFlowRobust: estimate the pose using iterative reweighted
 * least squares
 * @param prev_pts
 * @param prev_depth
 * @param next_pts
 * @param nodal_x
 * @param nodal_y
 * @param focal_x
 * @param focal_y
 * @param num_iters
 * @param translation
 * @param rotation
 */
Eigen::VectorXf getPoseFromFlowRobust(const std::vector<cv::Point2f>& prev_pts,
                           const std::vector<float>& prev_depth,
                           const std::vector<cv::Point2f>& next_pts,
                           float nodal_x, float nodal_y, float focal_x,
                           float focal_y, int num_iters,
                           std::vector<float>& translation,
                           std::vector<float>& rotation,
                           std::vector<int>& outliers);

void getPose2D(const std::vector<cv::Point2f*>& model_points,
               const std::vector<cv::Point2f*>& tracked_points, float& scale,
               float& angle);

// cv::Mat getPose3D(const std::vector<cv::Point3f*>& model_points,
//                  const std::vector<cv::Point3f*>& tracked_points,
//                  const std::vector<Status*>& points_status);

cv::Mat getRigidTransform(cv::Mat& a, cv::Mat& b);

cv::Mat getRigidTransform(cv::Mat& a, cv::Mat& b, std::vector<float>& cA,
                          std::vector<float>& cB);

void rotateBBox(const std::vector<cv::Point3f>& bBox, const cv::Mat& rotation,
                std::vector<cv::Point3f>& updatedBBox);

void rotatePoint(const cv::Point3f& point, const cv::Mat& rotation,
                 cv::Point3f& updatedPoint);

void rotatePoint(const cv::Vec3f& point, const cv::Mat& rotation,
                 cv::Vec3f& updatedPoint);

void rotatePoint(const cv::Vec3f& point, const cv::Mat& rotation,
                 cv::Point3f& updatedPoint);

void rotationVecToMat(const cv::Mat& vec, cv::Mat& mat);

class Pose{

public:

    Pose();

    /**
     * @brief Pose
     * @param r_mat 3x3 rotation matrix
     * @param t_vect rotation vector
     */
    Pose(cv::Mat& r_mat, cv::Mat& t_vect);

    /**
     * @brief Pose
     * @param pose eigen projection matrix 4x4
     */
    Pose(Eigen::Matrix4d& pose);

    /**
     * @brief Pose
     * @param beta 6x1 vector [tx,ty,tz,rx,ry,rz]
     */
    Pose(std::vector<double>& beta);

    /**
     * @brief Pose
     * @param beta 6x1 vector [tx,ty,tz,rx,ry,rz]
     */
    Pose(Eigen::VectorXf &beta);

    /**
     * @brief toCV pose to opencv rotation matrix and translation vector
     * @return a pair of Mat, [rotaion,translation]
     */
    std::pair<cv::Mat,cv::Mat> toCV() const;


    /**
     * @brief toEigen pose to eigen rotation matrix and translation vector
     * @return pair [matrix3d,vector3d>
     */
    std::pair<Eigen::Matrix3d,Eigen::Vector3d> toEigen() const;

    /**
     * @brief getBeta
     * @return 6x1 vector [tx,ty,tz,rx,ry,rz]
     */
    std::vector<double> getBeta();

    glm::mat4 toGL() const;

    Eigen::Matrix4d getPose(){return pose_;}

    void setPose(Eigen::Matrix4d& pose)
    {
        pose_ = pose;
    }

    void transform(Eigen::Matrix4d& transform);

    void transform(std::vector<double>& beta);

    std::vector<double> translation() const;

    Eigen::Quaternionf rotation() const;

    std::string str() const;

private:

    Eigen::MatrixX4d pose_;

    std::vector<double> init_beta_;
    cv::Mat init_rot, init_tr;

};

}  // end namespace

#endif  // POSE_ESTIMATION_H
