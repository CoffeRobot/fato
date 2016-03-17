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

#include "../include/VideoWriter.h"
#include "../include/filemanager.h"
#include <iostream>

using namespace std;
using namespace cv;

namespace fs = boost::filesystem;

namespace fato {

VideoWriter::VideoWriter(std::string path, string name, int width, int height,
                         int type, int framerate)
    : m_path(path),
      m_name(name),
      m_width(width),
      m_height(height),
      m_type(type),
      m_framerate(framerate),
      m_producerStopped(false),
      m_consumerStopped(false) {
  m_queueStatus = async(std::launch::async, &VideoWriter::initConsumer, this);
}

VideoWriter::~VideoWriter()
{
   stopRecording();
}

int VideoWriter::initConsumer() {

#ifdef VERBOSE_LOGGING
  cout << "Recorder thread initialized \n";
#endif
  createDir(m_path);

  string filename = m_path + m_name;

  string codec;
  if (m_type == IMG_TYPE::DEPTH)
    codec = "ffv1";
  else
    codec = "libx264";

  Encoderx264 writer(filename.c_str(), m_width, m_height, m_framerate,
                          codec.c_str(), 25, "superfast");

  bool savingFinished = false;

  int count = 0;

  while (!savingFinished) {
    Mat img;
    m_mutex.lock();
    if (m_imgsQueue.size() == 0 && m_producerStopped) {
      savingFinished = true;
      m_consumerStopped = true;
    } else if (m_imgsQueue.size() > 0) {
      img = m_imgsQueue.front();
      m_imgsQueue.pop();
    }
    m_mutex.unlock();
    if (img.data) {
      if (m_type == IMG_TYPE::DEPTH) {
        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(1);
        stringstream ss;
        ss << m_path << count << ".png";

        imwrite(ss.str(), img, compression_params);
      } else {
        writer.addFrame(img.data);
      }
      count++;
    }
  }

#ifdef VERBOSE_LOGGING
  cout << "VideoWriter finished!\n Frames written" << count << "\n\n\n";
#endif
  return 0;
}

void VideoWriter::write(cv::Mat img) {
  m_mutex.lock();
  m_imgsQueue.push(img);
  m_mutex.unlock();
}

bool VideoWriter::hasFinished() { return m_consumerStopped; }

}  // end namespace
