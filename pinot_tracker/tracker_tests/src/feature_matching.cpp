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
#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <algorithm>

#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/profiler.h"
#include "../../utilities/include/utilities.h"
#include "../../tracker/include/bounding_cube.h"
#include "../../utilities/include/visualization_ros.h"
#include "../../io/include/filemanager.h"
#include "../include/feature_matching.h"

using namespace cv;
using namespace std;

namespace pinot_tracker {

FeatureBenchmark::FeatureBenchmark()
{

}

FeatureBenchmark::~FeatureBenchmark()
{

}

void FeatureBenchmark::testVideo(string path)
{
  vector<string> image_names;
  vector<Rect> boxes;


  parseGT(path+"groundtruth.txt", boxes);
  getFiles(path+"imgs/", ".png", image_names);

  sort(image_names.begin(), image_names.end());

  cout << "GT " << boxes.size() << " " << image_names.size() << endl;



  for(auto i = 0; i < image_names.size(); ++i)
  {

    Mat img;
    img = imread(path+"imgs/" + image_names.at(i),1);

    if(i < boxes.size())
    {
      cout << boxes.at(i) << endl;
      rectangle(img, boxes.at(i), Scalar(0,0,255), 3);
    }

    imshow("benchmark", img);
    waitKey(30);
  }

}

void FeatureBenchmark::parseGT(string path, std::vector<Rect> &ground_truth) {
  ifstream file(path);

  if (!file.is_open())
  {
   cout << "Cannot open gt file!" << endl;

    return;
  }


  string line;
  while (getline(file, line)) {

    vector<string> words;
    boost::split(words, line, boost::is_any_of(","));


    if(words.size() == 4)
    {
      auto x = atof(words[0].c_str());
      auto y = atof(words[1].c_str());
      auto w = atof(words[2].c_str());
      auto h = atof(words[3].c_str());

      ground_truth.push_back(cv::Rect(Point2f(x,y), Point2f(x+w,y+h)));
    }
  }

  cout << "GT size " << ground_truth.size() << endl;
}

}  // end namespace

int main(int argc, char *argv[]) {
  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "pinot_tracker_feature_matching");

  vector<Rect> gt;

  pinot_tracker::FeatureBenchmark fb;
  fb.testVideo("/media/alessandro/Super "
               "Fat/Dataset/tracking_clean/ball/");

  return 0;
}
