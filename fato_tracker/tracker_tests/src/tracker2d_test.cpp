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

#include <boost/thread.hpp>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <chrono>
#include <draw_functions.h>
#include "../../tracker/include/tracker_2d.h"
#include "../../tracker/include/feature_matcher.hpp"

using namespace cv;
using namespace std;

cv::Point2d mouse_start_, mouse_end_;
bool is_mouse_dragging_, init_requested_, tracker_initialized_, img_updated_;

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
  auto set_point = [&](int x, int y) {
    if (x < mouse_start_.x) {
      mouse_end_.x = mouse_start_.x;
      mouse_start_.x = x;
    } else
      mouse_end_.x = x;

    if (y < mouse_start_.y) {
      mouse_end_.y = mouse_start_.y;
      mouse_start_.y = y;
    } else
      mouse_end_.y = y;
  };

  if (event == EVENT_LBUTTONDOWN) {
    mouse_start_.x = x;
    mouse_start_.y = y;
    mouse_end_ = mouse_start_;
    is_mouse_dragging_ = true;
    std::cout << "callback called! " << endl;
  } else if (event == EVENT_MOUSEMOVE && is_mouse_dragging_) {
    set_point(x, y);
  } else if (event == EVENT_LBUTTONUP) {
    set_point(x, y);
    is_mouse_dragging_ = false;
    init_requested_ = true;
  }
}


int main(int argc, char* argv[]) {
  cv::VideoCapture camera(1);

  fato::BriskMatcher brisk_matcher;
  Config params;

  std::unique_ptr<fato::FeatureMatcher> derived =
      std::unique_ptr<fato::BriskMatcher>(new fato::BriskMatcher);

  fato::Tracker tracker(params, 0, std::move(derived));

  namedWindow("Image Viewer");
  setMouseCallback("Image Viewer", mouseCallback, NULL);

  is_mouse_dragging_ = img_updated_ = init_requested_ = tracker_initialized_ =
      false;
  mouse_start_ = mouse_end_ = Point2f(0, 0);

  while (camera.isOpened()) {
    cv::Mat rgb_image_;
    camera >> rgb_image_;

    if (mouse_start_.x != mouse_end_.x && !tracker_initialized_) {
      rectangle(rgb_image_, mouse_start_, mouse_end_, Scalar(255, 0, 0), 3);
      img_updated_ = false;
    }
    if (!tracker_initialized_) {
      imshow("Image Viewer", rgb_image_);
      waitKey(1);
    }
    if (init_requested_) {
      tracker.init(rgb_image_, mouse_start_, mouse_end_);
      init_requested_ = false;
      tracker_initialized_ = true;
      waitKey(1);
    }

    if (tracker_initialized_) {
      Mat out;
      tracker.computeNext(rgb_image_);

      stringstream ss;
      Point2f p = tracker.getCentroid();
      circle(rgb_image_, p, 5, Scalar(255, 0, 0), -1);
      vector<Point2f> bbox = tracker.getBoundingBox();
      fato::drawBoundingBox(bbox, Scalar(255, 0, 0), 2, rgb_image_);
      imshow("Image Viewer", rgb_image_);
      waitKey(1);
    }
  }


  return 0;
}
