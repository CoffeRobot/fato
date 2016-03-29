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

#include "../include/target.hpp"
#include <iostream>
#include <fstream>

using namespace cv;

namespace fato {

void Target::init(std::vector<cv::Point3f> &points, cv::Mat &descriptors) {
  model_points_ = points;
  descriptors_ = descriptors.clone();

  centroid_ = Point3f(0, 0, 0);

  rel_distances_.reserve(model_points_.size());
  point_status_.resize(model_points_.size(), Status::LOST);

  for (auto pt : model_points_) {
    rel_distances_.push_back(pt - centroid_);
  }

  int reserved_memory = model_points_.size() / 10;

  active_to_model_.reserve(reserved_memory);
  active_points.reserve(reserved_memory);

  rotation = Mat(3, 3, CV_64FC1, 0.0f);
  translation = Mat(1, 3, CV_64FC1, 0.0f);
}

void Target::removeInvalidPoints(const std::vector<int> &ids) {



//  std::ofstream file;
//  file.open("/home/alessandro/debug/debug_match.txt",
//            std::ofstream::out | std::ofstream::app);

//  std::stringstream ss;

  int last_valid = active_points.size() - 1;

  //vector<int> ids_old = active_to_model_;

  for (auto id : ids) {
    int mid = active_to_model_.at(id);
    point_status_.at(mid) = Status::LOST;
  }

  vector<int> elem_ids(active_points.size(),0);
  for(int i = 0; i < elem_ids.size(); ++i)
      elem_ids[i] = i;


  for (auto id : ids) {

    auto& el_id = elem_ids.at(id);

    std::swap(active_points.at(el_id), active_points.at(last_valid));
    std::swap(active_to_model_.at(el_id), active_to_model_.at(last_valid));
    std::swap(elem_ids.at(el_id), elem_ids.at(last_valid));
    last_valid--;
  }

  int resized = last_valid + 1;
  active_points.resize(resized);
  active_to_model_.resize(resized);

//  ss << "Status after removing: " << std::endl;
//  for (int i = 0; i < point_status_.size(); ++i) {
//      auto s = point_status_.at(i);

//      if (s == Status::MATCH || s == Status::TRACK)
//          ss << "[" << i << "," << (int)s << "] ";
//  }
//  ss << "\n\n";

  if (!isConsistent()) {
    std::cout << "ERROR!" << std::endl;

//    ss << "Input List\n";
//    for(auto id : ids_old)
//    {
//        auto s = point_status_.at(id);
//        ss << "[" << id << "," << (int)s << "] ";
//    }
//    ss << "\n\n";
//    ss << "Updated List\n";
//    for(auto id : active_to_model_)
//    {
//        auto s = point_status_.at(id);
//        ss << "[" << id << "," << (int)s << "] ";
//    }
//    ss << "\n\n";
//    ss << "Status List\n";
//    for(auto i = 0; i < point_status_.size(); ++i)
//    {
//        bool in_remove = false;
//        bool in_active = false;

//        for(auto j : ids)
//        {
//            if(i == ids_old.at(j))
//            {
//                in_remove = true;
//                break;
//            }
//        }

//        for(auto j : active_to_model_)
//        {
//            if(i == j)
//            {
//                in_active = true;
//                break;
//            }
//        }

//        auto s = point_status_.at(i);
//        if(s == Status::MATCH || s == Status::TRACK)
//            ss << i << "," << (int)s << "," << in_active << "," << in_remove << "\n";
//    }
//    ss << "\n";
//    ss << "To remove: \n";
//    for(auto id : ids)
//    {
//        ss << old_active_to_model.at(id) << " ";
//    }
//    ss << "\n";
//    ss << "Resizing to: " << resized << "\n";
//    for(int i =0; i < old_active_to_model.size(); ++i)
//    {
//        if(i >= resized)
//          ss << "-";
//        else
//          ss << " ";

//        ss << old_active_to_model.at(i) << " " << (int)point_status_.at(old_active_to_model.at(i)) <<  "\n";
//    }
//    ss << "\n\n";

//    file << ss.str();

  }
 // file.close();
}

bool Target::isConsistent()
{
    for(auto i : active_to_model_)
    {
        if(point_status_.at(i) != Status::TRACK && point_status_.at(i) != Status::MATCH )
            return false;
    }
    return true;
}

}
