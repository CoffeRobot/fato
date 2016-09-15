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

#include "rigid_object_gl.h"

namespace rendering {

RigidObject::RigidObject(std::string model, std::string ver_shader,
                         std::string frag_shader)
    : model((GLchar*)model.c_str()),
      shader((GLchar*)ver_shader.c_str(), (GLchar*)frag_shader.c_str()),
      model_matrix_(),
      is_visible_(true) {
  computeBoundingBox();
}

RigidObject::~RigidObject() {}

void RigidObject::setVisible(bool is_visible) { is_visible_ = is_visible; }

bool RigidObject::isVisible() { return is_visible_; }

void RigidObject::updatePose(Eigen::Transform<double, 3, Eigen::Affine>& pose) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      model_matrix_[j][i] = pose(i, j);
    }
    model_matrix_[3][i] = pose(i, 3);
  }
  model_matrix_[3][3] = 1;

}

void RigidObject::updatePose(glm::mat4 &pose)
{
    model_matrix_ = pose;
}

vector<float> RigidObject::getBoundingBox() { return bounding_box_; }

void RigidObject::computeBoundingBox() {
  mins_ = glm::vec3(numeric_limits<float>::max());
  maxs_ = glm::vec3(-numeric_limits<float>::max());

  for (int i = 0; i < model.getMeshCount(); ++i) {
    const Mesh& m = model.getMesh(i);

    for (const Vertex& v : m.vertices_) {
      const glm::vec3& pos = v.Position;

      if (pos.x < mins_.x) mins_.x = pos.x;
      if (pos.y < mins_.y) mins_.y = pos.y;
      if (pos.z < mins_.z) mins_.z = pos.z;

      if (pos.x > maxs_.x) maxs_.x = pos.x;
      if (pos.y > maxs_.y) maxs_.y = pos.y;
      if (pos.z > maxs_.z) maxs_.z = pos.z;
    }
  }
  // 8 3d points -> 24
  bounding_box_.resize(24, 0.0f);

  bounding_box_[0] = mins_.x;
  bounding_box_[1] = mins_.y;
  bounding_box_[2] = mins_.z;

  bounding_box_[3] = maxs_.x;
  bounding_box_[4] = mins_.y;
  bounding_box_[5] = mins_.z;

  bounding_box_[6] = maxs_.x;
  bounding_box_[7] = maxs_.y;
  bounding_box_[8] = mins_.z;

  bounding_box_[9] = mins_.x;
  bounding_box_[10] = maxs_.y;
  bounding_box_[11] = mins_.z;

  bounding_box_[12] = mins_.x;
  bounding_box_[13] = mins_.y;
  bounding_box_[14] = maxs_.z;

  bounding_box_[15] = maxs_.x;
  bounding_box_[16] = mins_.y;
  bounding_box_[17] = maxs_.z;

  bounding_box_[18] = maxs_.x;
  bounding_box_[19] = maxs_.y;
  bounding_box_[20] = maxs_.z;

  bounding_box_[21] = mins_.x;
  bounding_box_[22] = maxs_.y;
  bounding_box_[23] = maxs_.z;
}

}  // end namespace
