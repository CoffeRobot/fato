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
#include <iomanip>

using namespace std;
using namespace cv;


float mouse_x, mouse_y;

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    cout << "Left button of the mouse is clicked - position (" << x << ", " << y
         << ")" << endl;
  } else if (event == EVENT_RBUTTONDOWN) {
    cout << "Right button of the mouse is clicked - position (" << x << ", "
         << y << ")" << endl;
  } else if (event == EVENT_MBUTTONDOWN) {
    cout << "Middle button of the mouse is clicked - position (" << x << ", "
         << y << ")" << endl;
  } else if (event == EVENT_MOUSEMOVE) {
    cout << "Mouse move over the window - position (" << x << ", " << y << ")"
         << endl;
    mouse_x = x;
    mouse_y = y;
  }
}

void initKalman() {}

int main(int argc, char** argv) {
  KalmanFilter KF(4, 2, 0);
  KF.transitionMatrix =
      *(Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

  Mat_<float> measurement(2, 1);
  measurement.setTo(Scalar(0));

  // init...
  KF.statePre.at<float>(0) = mouse_x;
  KF.statePre.at<float>(1) = mouse_y;
  KF.statePre.at<float>(2) = 0;
  KF.statePre.at<float>(3) = 0;
  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(.1));

  cout << fixed << setprecision(2) << "process noise cov \n"
        << KF.processNoiseCov << " \n"
        << "measurementNoiseCov \n"
        << KF.measurementNoiseCov << "\n"
        << "errorCovPost \n"
        << KF.errorCovPost << endl;

  return 0;

  // Create a window
  namedWindow("Kalman tutorial", 1);

  // set the callback function for any mouse event
  setMouseCallback("Kalman tutorial", CallBackFunc, NULL);

  Mat res_image(480,640, CV_8UC3, Scalar(0,0,0));

  mouse_x = mouse_y = 0;


  vector<Point> mouse_line;
  vector<Point> estimated_line;


  bool running = true;
  while(running)
  {

    Mat res_image(480,640, CV_8UC3, Scalar(0,0,0));

    // First predict, to update the internal statePre variable
    Mat prediction = KF.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

    // Get mouse point
    measurement(0) = mouse_x;
    measurement(1) = mouse_y;

    Point measPt(measurement(0),measurement(1));

    // The "correct" phase that is going to use the predicted value and our measurement
    Mat estimated = KF.correct(measurement);
    Point statePt(estimated.at<float>(0),estimated.at<float>(1));

    mouse_line.push_back(measPt);
    estimated_line.push_back(statePt);

    if(mouse_line.size() > 0)
    {
        polylines(res_image, mouse_line, false, Scalar(255,0,0));
        polylines(res_image, estimated_line, false, Scalar(0,255,0));
    }

    //circle(res_image, measPt, 1, Scalar(255,0,0), -1);
    //circle(res_image, statePt, 1, Scalar(0,255,0), -1);

    imshow("Kalman tutorial",res_image);

    auto c = waitKey(100);

    if(c == 'q')
      break;

  }


  return 0;
}
