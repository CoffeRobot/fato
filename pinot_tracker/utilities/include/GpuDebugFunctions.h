#pragma once
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <random>
#include <fstream>
#include <memory>
#include <fstream>

#include "Constants.h"

namespace pinot_tracker {

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
