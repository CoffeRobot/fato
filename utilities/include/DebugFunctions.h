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

#ifndef DEBUG_FUNCTIONS_H
#define DEBUG_FUNCTIONS_H

#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <random>
#ifdef __unix__
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#else
#include <Eigen/Dense>
#include <Eigen/Geometry>
#endif
#include <fstream>

#include "constants.h"
#include "ToString.h"

namespace fato{

void cross(const cv::Point2f& p, const cv::Scalar& c, int width, cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::KeyPoint>& keypoints,
                       std::vector<cv::Point2f>& points,
                       const std::vector<bool>& clustered,
                       const std::vector<bool>& border,
                       const std::vector<FatoStatus>& status, bool drawLines,
                       bool drawFalse, cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::Point3f>& keypoints,
                       std::vector<cv::Point3f>& points,
                       const std::vector<bool>& clustered,
                       const std::vector<bool>& border,
                       const std::vector<FatoStatus>& status, bool drawLines,
                       bool drawFalse, const float focal,
                       const cv::Point2f& center, std::ofstream& file,
                       cv::Mat& out);



void drawCentroidVotes(const std::vector<cv::Point3f*>& keypoints,
                       const std::vector<cv::Point3f>& votes,
                       const std::vector<bool>& clustered,
                       const std::vector<bool>& border,
                       const std::vector<FatoStatus*>& status, bool drawLines,
                       bool drawFalse, const float focal,
                       const cv::Point2f& center, cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::Point3f*>& keypoints,
                       const std::vector<cv::Point3f>& votes,
                       const std::vector<bool>& clustered,
                       const std::vector<bool>& border,
                       const std::vector<FatoStatus*>& status, bool drawLines,
                       bool drawFalse, const float focal,
                       const cv::Point2f& center, std::ofstream& file,
                       cv::Mat& out);

void buildCompositeImg(const cv::Mat& fst, const cv::Mat& scd, cv::Mat& out);

void drawObjectLocation(const cv::Point2f& fstC,
                        const std::vector<cv::Point2f>& fstBBox,
                        const cv::Point2f& scdC,
                        const std::vector<cv::Point2f>& scdBBox, cv::Mat& out);

void drawObjectLocation(const cv::Point3f& fstC,
                        const std::vector<cv::Point3f>& fstBBox,
                        const cv::Point3f& scdC,
                        const std::vector<cv::Point3f>& scdBBox,
                        const float focal, const cv::Point2f& center,
                        cv::Mat& out);

/*void drawObjectLocation(const BorgCube& fstCube, const BorgCube& updCube,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out);

void drawObjectLocation(const BorgCube& updCube,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out);
*/

void drawKeypointsMatching(const std::vector<cv::KeyPoint>& fstPoint,
                           const std::vector<cv::KeyPoint>& scdPoints,
                           const std::vector<FatoStatus>& pointStatus,
                           const std::vector<cv::Scalar>& colors, int& numMatch,
                           int& numTrack, int& numBoth, bool drawLines,
                           cv::Mat& out);

void drawPointsMatching(const std::vector<cv::Point3f>& fstPoints,
                        const std::vector<cv::Point3f>& scdPoints,
                        const std::vector<FatoStatus>& pointStatus,
                        const std::vector<cv::Scalar>& colors, int& numMatch,
                        int& numTrack, int& numBoth, bool drawLines,
                        const float focal, const cv::Point2f& center,
                        cv::Mat& out);

void drawPointsMatching(const std::vector<cv::Point3f*>& fstPoints,
                        const std::vector<cv::Point3f*>& scdPoints,
                        const std::vector<FatoStatus*>& pointStatus,
                        const std::vector<cv::Scalar*>& colors, int& numMatch,
                        int& numTrack, int& numBoth, bool drawLines,
                        const float focal, const cv::Point2f& center,
                        cv::Mat& out);

void drawPointsMatchingICRA(const std::vector<cv::Point3f*>& fstPoints,
                            const std::vector<cv::Point3f*>& scdPoints,
                            const std::vector<FatoStatus*>& pointStatus,
                            const std::vector<cv::Scalar*>& colors,
                            int& numMatch, int& numTrack, int& numBoth,
                            bool drawLines, const float focal,
                            const cv::Point2f& center, cv::Mat& out);

void countKeypointsMatching(const std::vector<cv::KeyPoint>& fstPoint,
                            const std::vector<cv::KeyPoint>& scdPoints,
                            const std::vector<FatoStatus>& pointStatus,
                            int& numMatch, int& numTrack, int& numBoth);

void countKeypointsMatching(const std::vector<FatoStatus*>& pointStatus,
                            int& numMatch, int& numTrack, int& numBoth);

void drawKeipointsStats(const int init, const int matched, const int tracked,
                        const int both, cv::Mat& out);

void drawInformationHeader(const int numFrames, const float scale,
                           const float angle, int clusterSize, int matched,
                           int tracked, cv::Mat& out);

void drawInformationHeaderICRA(cv::Point2f& top, const std::string frame,
                               const std::string angle,
                               const std::string visibility, float alpha,
                               int width, int height, cv::Mat& out);

void drawTriangle(const cv::Point2f& a, const cv::Point2f& b,
                  const cv::Point2f& c, cv::Scalar color, float alpha,
                  cv::Mat& out);

void drawTriangleMask(const cv::Point2f& a, const cv::Point2f& b,
                      const cv::Point2f& c, cv::Mat1b& out);

void debugCalculations(std::string path,
                       const std::vector<cv::KeyPoint>& currentKps,
                       const std::vector<cv::KeyPoint>& fstKps);

std::string faceToString(int face);

cv::Point2f reprojectPoint(const float focal, const cv::Point2f& center,
                           const cv::Point3f& src);

bool reprojectPoint(const float focal, const cv::Point2f& center,
                    const cv::Point3f& src, cv::Point2f& dst);


void drawVotes(const std::vector<cv::Point2f>& votes, cv::Scalar& color,
               cv::Mat& out);



} // end namesapce


#endif
