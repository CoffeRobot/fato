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

#include "../include/visualization_ros.h"

namespace pinot_tracker {

void getCubeMarker(const std::vector<cv::Point3f>& front_points,
                   const std::vector<cv::Point3f>& back_points,
                   std::vector<visualization_msgs::Marker>& faces) {
  // front face
  visualization_msgs::Marker front_face;
  front_face.ns = "cube_front";
  front_face.id = 1;

  front_face.color.r = 0.0f;
  front_face.color.g = 0.0f;
  front_face.color.b = 1.0f;
  front_face.color.a = 1.0f;

  getFaceMarker(front_points.at(0), front_points.at(1), front_points.at(2),
                front_points.at(3), front_face);

  visualization_msgs::Marker back_face;
  back_face.ns = "cube_back";
  back_face.id = 2;

  back_face.color.r = 1.0f;
  back_face.color.g = 0.0f;
  back_face.color.b = 0.0f;
  back_face.color.a = 1.0f;

  getFaceMarker(back_points.at(0), back_points.at(1), back_points.at(2),
                back_points.at(3), back_face);

  visualization_msgs::Marker left_face;
  left_face.ns = "cube_left";
  left_face.id = 3;

  left_face.color.r = 0.0f;
  left_face.color.g = 1.0f;
  left_face.color.b = 0.0f;
  left_face.color.a = 1.0f;

  getFaceMarker(front_points.at(1), back_points.at(1), back_points.at(2),
                front_points.at(2), left_face);

  visualization_msgs::Marker right_face;
  right_face.ns = "cube_right";
  right_face.id = 4;

  right_face.color.r = 1.0f;
  right_face.color.g = 1.0f;
  right_face.color.b = 0.0f;
  right_face.color.b = 1.0f;

  getFaceMarker(front_points.at(0), back_points.at(0), back_points.at(3),
                front_points.at(3), right_face);

  visualization_msgs::Marker top_face;
  top_face.ns = "cube_top";
  top_face.id = 5;

  top_face.color.r = 0.0f;
  top_face.color.g = 1.0f;
  top_face.color.b = 1.0f;
  top_face.color.a = 1.0f;

  getFaceMarker(front_points.at(0), back_points.at(0), back_points.at(1),
                front_points.at(1), top_face);

  visualization_msgs::Marker down_face;
  down_face.ns = "cube_down";
  down_face.id = 6;

  down_face.color.r = 1.0f;
  down_face.color.g = 0.0f;
  down_face.color.b = 1.0f;
  down_face.color.a = 1.0;

  getFaceMarker(front_points.at(3), back_points.at(3), back_points.at(2),
                front_points.at(2), down_face);

  faces.push_back(front_face);
  faces.push_back(back_face);
  faces.push_back(left_face);
  faces.push_back(right_face);
  faces.push_back(top_face);
  faces.push_back(down_face);
}

void getFaceMarker(cv::Point3f a, cv::Point3f b, cv::Point3f c, cv::Point3f d,
                   visualization_msgs::Marker &face) {
  face.header.frame_id = "camera_rgb_optical_frame";
  face.type = visualization_msgs::Marker::LINE_STRIP;
  face.action = visualization_msgs::Marker::ADD;
  face.scale.x = 0.01;

  geometry_msgs::Point p, tmp;

  p.x = a.x;
  p.y = -a.y;
  p.z = a.z;
  face.points.push_back(p);
  tmp = p;

  p.x = b.x;
  p.y = -b.y;
  p.z = b.z;
  face.points.push_back(p);

  p.x = c.x;
  p.y = -c.y;
  p.z = c.z;
  face.points.push_back(p);

  p.x = d.x;
  p.y = -d.y;
  p.z = d.z;
  face.points.push_back(p);
  face.points.push_back(tmp);
}
}
