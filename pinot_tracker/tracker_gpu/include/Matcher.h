#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <limits>

class Matcher
{
public:
	Matcher();
	~Matcher();

	void match(const cv::Mat& query, const cv::Mat& train, int bestNum, 
		       std::vector<std::vector<cv::DMatch> >& matches);

	void match32(const cv::Mat& query, const cv::Mat& train, int bestNum,
		std::vector<std::vector<cv::DMatch>>& matches);


};

