#include "../include/matcher.h"
#include <list>

using namespace std;
using namespace cv;

#ifdef __APPLE__
typedef long long __int64;
#include <immintrin.h>
#elif __unix__
typedef long long __int64;
typedef int32_t __int32;
#include <smmintrin.h>
#endif

namespace pinot_tracker
{


Matcher::Matcher()
{
}


Matcher::~Matcher()
{
}

void Matcher::match(const Mat& query, const Mat& train, int bestNum, vector<vector<DMatch>>& matches)
{

	__int64* fstData = (__int64*)query.data;
	__int64* scdData = (__int64*)train.data;

    // TODO(alessandro.pieropan@gmail.com): can be optimized to make it faster
	for (size_t i = 0; i < query.rows; i++)
	{

		DMatch tmp;
		tmp.distance = numeric_limits<float>::max();
		tmp.queryIdx = i;
		tmp.trainIdx = -1;

		list<DMatch> bestMatches(bestNum, DMatch(i, -1, numeric_limits<float>::max()));

		for (size_t j = 0; j < train.rows; j++)
		{
			int fstIdx = 8 * i;
			int scdIdx = 8 * j;

			int distance = 0;

			for (size_t k = 0; k < 8; k++)
			{
				distance += _mm_popcnt_u64(fstData[fstIdx + k] ^ scdData[scdIdx + k]);
			}

			for (auto it = bestMatches.begin(); it != bestMatches.end(); ++it)
			{
				if (distance < (*it).distance)
				{
					bestMatches.insert(it, DMatch(i, j, distance));
					bestMatches.pop_back();
					break;
				}
			}
		}

		matches.push_back(vector<DMatch>(bestMatches.begin(), bestMatches.end()));
	}
}


void Matcher::match32(const cv::Mat& query, const cv::Mat& train, int bestNum,
	std::vector<std::vector<cv::DMatch>>& matches)
{
	__int32* fstData = (__int32*)query.data;
	__int32* scdData = (__int32*)train.data;


    // TODO(alessandro.pieropan@gmail.com): can be optimized to make it faster
	for (size_t i = 0; i < query.rows; i++)
	{

		DMatch tmp;
		tmp.distance = numeric_limits<float>::max();
		tmp.queryIdx = i;
		tmp.trainIdx = -1;

		list<DMatch> bestMatches(bestNum, DMatch(i, -1, numeric_limits<float>::max()));

		for (size_t j = 0; j < train.rows; j++)
		{
			int fstIdx = 8 * i;
			int scdIdx = 8 * j;

			int distance = 0;

			for (size_t k = 0; k < 4; k++)
			{
				distance += _mm_popcnt_u64(fstData[fstIdx + k] ^ scdData[scdIdx + k]);
			}

			for (auto it = bestMatches.begin(); it != bestMatches.end(); ++it)
			{
				if (distance < (*it).distance)
				{
					bestMatches.insert(it, DMatch(i, j, distance));
					bestMatches.pop_back();
					break;
				}
			}
		}

		matches.push_back(vector<DMatch>(bestMatches.begin(), bestMatches.end()));
	}
}

} // end namespace
