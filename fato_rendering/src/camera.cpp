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
#include "camera.h"
#include <iostream>

namespace rendering {

// Constructor with vectors
Camera::Camera(glm::vec3 position, glm::vec3 up, GLfloat yaw, GLfloat pitch)
    : front_(glm::vec3(0.0f, 0.0f, -1.0f)),
      MovementSpeed(SPEED),
      MouseSensitivity(SENSITIVTY),
      Zoom(ZOOM) {
  position_ = position;
  world_up_ = up;
  yaw_ = yaw;
  pitch_ = pitch;
  updateCameraVectors();
  camera_matrix_ = glm::lookAt(position_, position_ + front_, up_);

  glm::vec3 f = position_ + front_;
  std::cout << "camera frame" << std::endl;
  std::cout << position_.x << " " << position_.y << " " << position_.z << std::endl;
  std::cout << f.x << " " << f.y << " " << f.z << std::endl;
  std::cout << up_.x << " " << up_.y << " " << up_.z << std::endl;
}
// Constructor with scalar values
Camera::Camera(GLfloat posX, GLfloat posY, GLfloat posZ, GLfloat upX,
               GLfloat upY, GLfloat upZ, GLfloat yaw, GLfloat pitch)
    : front_(glm::vec3(0.0f, 0.0f, -1.0f)),
      MovementSpeed(SPEED),
      MouseSensitivity(SENSITIVTY),
      Zoom(ZOOM) {
  position_ = glm::vec3(posX, posY, posZ);
  world_up_ = glm::vec3(upX, upY, upZ);
  yaw_ = yaw;
  pitch_ = pitch;
  updateCameraVectors();
  camera_matrix_ = glm::lookAt(position_, position_ + front_, up_);

  glm::vec3 f = position_ + front_;
  std::cout << "camera frame" << std::endl;
  std::cout << position_.x << " " << position_.y << " " << position_.z << std::endl;
  std::cout << f.x << " " << f.y << " " << f.z << std::endl;
  std::cout << up_.x << " " << up_.y << " " << up_.z << std::endl;
}

Camera::Camera(glm::vec3 position, glm::vec3 up, glm::vec3 front)
{
    camera_matrix_ = glm::lookAt(position, front, up);
}

glm::mat4 Camera::GetViewMatrix() {
  return camera_matrix_;
}

void Camera::ProcessKeyboard(Camera_Movement direction, GLfloat deltaTime) {
  GLfloat velocity = MovementSpeed * deltaTime;
  if (direction == FORWARD) position_ += front_ * velocity;
  if (direction == BACKWARD) position_ -= front_ * velocity;
  if (direction == LEFT) position_ -= right_ * velocity;
  if (direction == RIGHT) position_ += right_ * velocity;
}

void Camera::ProcessMouseMovement(GLfloat xoffset, GLfloat yoffset,
                                  GLboolean constrainPitch) {
  xoffset *= this->MouseSensitivity;
  yoffset *= this->MouseSensitivity;

  this->yaw_ += xoffset;
  this->pitch_ += yoffset;

  // Make sure that when pitch is out of bounds, screen doesn't get flipped
  if (constrainPitch) {
    if (this->pitch_ > 89.0f) this->pitch_ = 89.0f;
    if (this->pitch_ < -89.0f) this->pitch_ = -89.0f;
  }

  // Update Front, Right and Up Vectors using the updated Eular angles
  this->updateCameraVectors();
}

void Camera::ProcessMouseScroll(GLfloat yoffset) {
  if (Zoom >= 1.0f && Zoom <= 45.0f) Zoom -= yoffset;
  if (Zoom <= 1.0f) Zoom = 1.0f;
  if (Zoom >= 45.0f) Zoom = 45.0f;
}

void Camera::updateCameraVectors() {
  // Calculate the new Front vector
  glm::vec3 front;
  front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
  front.y = sin(glm::radians(pitch_));
  front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
  front_ = glm::normalize(front);
  // Also re-calculate the Right and Up vector
  right_ = glm::normalize(glm::cross(
      front_, world_up_));  // Normalize the vectors, because their length gets
                            // closer to 0 the more you look up or down which
                            // results in slower movement.
  up_ = glm::normalize(glm::cross(right_, front_));
}
}
