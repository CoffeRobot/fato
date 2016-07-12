/*****************************************************************************/
/*  Copyright (c) 2016, Alessandro Pieropan                                  */
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

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>b
#include <random>
#include <cmath>


using namespace std;
using namespace cv;
using namespace Eigen;

double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}



template <typename T>
string toString(Mat& mat) {
  stringstream ss;
  ss << fixed << setprecision(3);
  for (auto i = 0; i < mat.rows; ++i) {
    for (auto j = 0; j < mat.cols; ++j) {
      ss << mat.at<T>(i, j) << " ";
    }
    ss << "\n";
  }

  return ss.str();
}

void printKalman(KalmanFilter& kf) {
  ofstream file("/home/alessandro/debug/pkf.txt");

  Mat prediction = kf.predict();

  file << "transition matrix: " << toString<float>(kf.transitionMatrix) << "\n"
       << "post " << toString<float>(kf.statePost) << "\n"
       << "predict " << toString<float>(prediction) << endl;

  Mat_<float> measurement(6, 1);
  measurement(0) = 0.01;
  measurement(1) = 0;
  measurement(2) = 0;
  measurement(3) = 0;
  measurement(4) = 0;
  measurement(5) = 0;

  Mat estimated = kf.correct(measurement);

  file << "measurements: " << toString<float>(measurement) << "\n"
       << "post " << toString<float>(kf.statePost) << "\n"
       << "estimated " << toString<float>(estimated) << endl;

  file.close();

}

KalmanFilter initKalmanPose() {
  KalmanFilter kalman_pose_pnp_(18, 6);

  Mat A = Mat::eye(18, 18, CV_32FC1);

  double dt = 1;

  for (auto i = 0; i < 9; ++i) {
    auto id_vel = i + 3;
    auto id_acc = i + 6;
    auto id_vel2 = i + 12;
    auto id_acc2 = i + 15;

    if (id_vel < 9) A.at<float>(i, id_vel) = dt;
    if (id_acc < 9) A.at<float>(i, id_acc) = 0.5 * dt * dt;
    if (id_vel2 < 18) A.at<float>(i + 9, id_vel2) = dt;
    if (id_acc2 < 18) A.at<float>(i + 9, id_acc2) = 0.5 * dt * dt;
  }

  kalman_pose_pnp_.transitionMatrix = A.clone();

  kalman_pose_pnp_.measurementMatrix.at<float>(0, 0) = 1;
  kalman_pose_pnp_.measurementMatrix.at<float>(1, 1) = 1;
  kalman_pose_pnp_.measurementMatrix.at<float>(2, 2) = 1;
  kalman_pose_pnp_.measurementMatrix.at<float>(3, 9) = 1;
  kalman_pose_pnp_.measurementMatrix.at<float>(4, 10) = 1;
  kalman_pose_pnp_.measurementMatrix.at<float>(5, 11) = 1;

  setIdentity(kalman_pose_pnp_.processNoiseCov, Scalar::all(1e-1));
  setIdentity(kalman_pose_pnp_.measurementNoiseCov, Scalar::all(1e-2));
  setIdentity(kalman_pose_pnp_.errorCovPost, Scalar::all(.1));

  kalman_pose_pnp_.measurementNoiseCov.at<float>(0, 0) = 5;
  kalman_pose_pnp_.measurementNoiseCov.at<float>(1, 1) = 1;
  kalman_pose_pnp_.measurementNoiseCov.at<float>(2, 2) = 1;
  kalman_pose_pnp_.measurementNoiseCov.at<float>(3, 3) = 1;
  kalman_pose_pnp_.measurementNoiseCov.at<float>(4, 4) = 1;
  kalman_pose_pnp_.measurementNoiseCov.at<float>(5, 5) = 1;

  kalman_pose_pnp_.statePre.at<float>(0) = 0;
  kalman_pose_pnp_.statePre.at<float>(1) = 0;
  kalman_pose_pnp_.statePre.at<float>(2) = 0;
  kalman_pose_pnp_.statePre.at<float>(3) = 0;
  kalman_pose_pnp_.statePre.at<float>(4) = 0;
  kalman_pose_pnp_.statePre.at<float>(5) = 0;
  kalman_pose_pnp_.statePre.at<float>(6) = 0;
  kalman_pose_pnp_.statePre.at<float>(7) = 0;
  kalman_pose_pnp_.statePre.at<float>(8) = 0;
  kalman_pose_pnp_.statePre.at<float>(9) = 0;
  kalman_pose_pnp_.statePre.at<float>(10) = 0;
  kalman_pose_pnp_.statePre.at<float>(11) = 0;
  kalman_pose_pnp_.statePre.at<float>(12) = 0;
  kalman_pose_pnp_.statePre.at<float>(13) = 0;
  kalman_pose_pnp_.statePre.at<float>(14) = 0;
  kalman_pose_pnp_.statePre.at<float>(15) = 0;
  kalman_pose_pnp_.statePre.at<float>(16) = 0;
  kalman_pose_pnp_.statePre.at<float>(17) = 0;

  return kalman_pose_pnp_;
}

void testKalmanPose()
{

    ofstream file("/home/alessandro/debug/pkf.txt");
    random_device rd;
    default_random_engine dre(rd());

    std::uniform_real_distribution<float> dist(0, 1);

    float tx = 1;
    float tx_acc =0;
    float rx_acc = 0;
    float rx = deg2rad(2);

    uniform_real_distribution<float> deg_dist(0, rx/4);

    KalmanFilter kfp = initKalmanPose();

    file << toString<float>(kfp.measurementNoiseCov) << "\n";

    for(auto i = 0; i < 5; ++i)
    {
        float val = dist(dre);

        tx_acc += val + tx;
        rx_acc += rx + deg_dist(dre);

        Mat_<float> measurement(6, 1);
        measurement(0) = tx_acc;
        measurement(1) = 0;
        measurement(2) = 0;
        measurement(3) = 0;
        measurement(4) = 0;
        measurement(5) = rx_acc;

        Mat predictions = kfp.predict();
        Mat corrections = kfp.correct(measurement);
        Mat pred_t, mea_t, corr_t;
        transpose(predictions, pred_t);
        transpose(corrections, corr_t);
        transpose(measurement, mea_t);

        file << "predictions \n" << toString<float>(pred_t);
        file << "measurements \n" << toString<float>(mea_t);
        file << "corrections \n" << toString<float>(corr_t) << "\n";
    }

    file.close();

}

void test_eigen() {
  Matrix3f m;

  m(0,0) = 1;
  m(0,1) = -0.001;
  m(0,2) = 0.006;

  m(1,0) = 0.001;
  m(1,1) = 1;
  m(1,2) = -0.001;

  m(2,0) = -0.006;
  m(2,1) = 0.001;
  m(2,2) = 1;


  cout << "original rotation:" << endl;
  cout << m << endl << endl;

  Vector3f ea = m.eulerAngles(2, 1, 0);
  cout << "to Euler angles:" << endl;
  cout << ea << endl << endl;

  Matrix3f n;
  n = AngleAxisf(ea[0], Vector3f::UnitZ()) *
      AngleAxisf(ea[1], Vector3f::UnitY()) *
      AngleAxisf(ea[2], Vector3f::UnitX());
  cout << "recalc original rotation:" << endl;
  cout << n << endl;
}

float mouse_x, mouse_y;

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    //cout << "Left button of the mouse is clicked - position (" << x << ", " << y
    //     << ")" << endl;
  } else if (event == EVENT_RBUTTONDOWN) {
    //cout << "Right button of the mouse is clicked - position (" << x << ", "
    //     << y << ")" << endl;
  } else if (event == EVENT_MBUTTONDOWN) {
    //cout << "Middle button of the mouse is clicked - position (" << x << ", "
    //     << y << ")" << endl;
  } else if (event == EVENT_MOUSEMOVE) {
//    cout << "Mouse move over the window - position (" << x << ", " << y << ")"
//         << endl;
    mouse_x = x;
    mouse_y = y;
  }
}

int main(int argc, char** argv) {

    //testKalmanPose();

    //test_eigen();

  KalmanFilter KF(4, 2, 0);
  KF.transitionMatrix =
      *(Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

  Mat_<float> measurement(2, 1);
  measurement.setTo(Scalar(0));

  // init...
  KF.statePre.at<float>(0) = 320;
  KF.statePre.at<float>(1) = 240;
  KF.statePre.at<float>(2) = 0;
  KF.statePre.at<float>(3) = 0;
  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(.1));

  cout << fixed << setprecision(2) << "process noise cov \n"
       << KF.processNoiseCov << " \n"
       << "measurementNoiseCov \n" << KF.measurementNoiseCov << "\n"
       << "errorCovPost \n" << KF.errorCovPost << endl;

  return 0;

  // Create a window
  namedWindow("Kalman tutorial", 1);

  // set the callback function for any mouse event
  setMouseCallback("Kalman tutorial", CallBackFunc, NULL);

  Mat res_image(480, 640, CV_8UC3, Scalar(0, 0, 0));

  mouse_x = 320;
  mouse_y = 240;

  vector<Point> mouse_line;
  vector<Point> estimated_line;

  bool running = true;
  while (running) {
    Mat res_image(480, 640, CV_8UC3, Scalar(0, 0, 0));

    // First predict, to update the internal statePre variable
    Mat prediction = KF.predict();
    Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

    // Get mouse point
    measurement(0) = mouse_x;
    measurement(1) = mouse_y;

    Point measPt(measurement(0), measurement(1));

    // The "correct" phase that is going to use the predicted value and our
    // measurement
    Mat estimated = KF.correct(measurement);
    Point statePt(estimated.at<float>(0), estimated.at<float>(1));

    mouse_line.push_back(measPt);
    estimated_line.push_back(statePt);

    if (mouse_line.size() > 0) {
      polylines(res_image, mouse_line, false, Scalar(255, 0, 0));
      polylines(res_image, estimated_line, false, Scalar(0, 255, 0));
    }

    // circle(res_image, measPt, 1, Scalar(255,0,0), -1);
    // circle(res_image, statePt, 1, Scalar(0,255,0), -1);

    imshow("Kalman tutorial", res_image);

    auto c = waitKey(100);

    if (c == 'q') break;
  }

  return 0;
}
