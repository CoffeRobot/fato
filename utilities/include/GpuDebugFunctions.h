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

#ifndef GPU_DEBUG_H
#define GPU_DEBUG_H


#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <random>
#include <fstream>
#include <memory>
#include <fstream>

#include "constants.h"

namespace fato {

void DrawFlowPoints(const std::vector<cv::Point2f>* points,
                    const std::vector<Status>* pointsStatus,
                    const std::vector<int>* pointsIds, cv::Mat& out);

void DrawDetectedPoints(const std::vector<cv::Point2f>* initPts,
                        const std::vector<cv::Point2f>* updPts,
                        const std::vector<Status>* ptsStatus,
                        const std::vector<int>* ptsIds, cv::Mat& out);

void drawVotesGPU(const std::vector<cv::Point2f>* points,
                  const std::vector<Status>* pointsStatus,
                  const std::vector<cv::Point2f>* votes,
                  const std::vector<int>* pointsIds, cv::Mat& out);

void printPointsStatus(const std::vector<cv::Point2f>* points,
                       const std::vector<Status>* pointsStatus,
                       const std::vector<cv::Point2f>* votes,
                       const std::vector<int>* pointsIds, std::ofstream& file);

}  // end namespace

#endif
