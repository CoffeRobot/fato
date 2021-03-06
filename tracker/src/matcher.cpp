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

#include "../include/matcher.h"
#include <list>

using namespace std;
using namespace cv;

#ifdef __APPLE__
typedef long long __int64;
#include <immintrin.h>
#elif __arm__
typedef long long __int64;
typedef int __int32;
#elif __unix__
typedef long long __int64;
typedef int32_t __int32;
#include <smmintrin.h>
#endif

namespace fato
{


CustomMatcher::CustomMatcher()
{
}


CustomMatcher::~CustomMatcher()
{
}


void CustomMatcher::match(const Mat& query, const Mat& train, int bestNum, vector<vector<DMatch>>& matches)
{

  

	__int64* fstData = (__int64*)query.data;
	__int64* scdData = (__int64*)train.data;

    matches.reserve(query.rows);

    // TODO(alessandro.pieropan@gmail.com): can be optimized to make it faster
    for (size_t i = 0; i < query.rows; ++i)
	{

		list<DMatch> bestMatches(bestNum, DMatch(i, -1, numeric_limits<float>::max()));

        for (size_t j = 0; j < train.rows; ++j)
		{
			int fstIdx = 8 * i;
			int scdIdx = 8 * j;

			int distance = 0;

            for (size_t k = 0; k < 8; ++k)
			{
#ifdef __arm__
		distance += __builtin_popcountll(fstData[fstIdx + k] ^ scdData[scdIdx + k]);
#else
		distance += _mm_popcnt_u64(fstData[fstIdx + k] ^ scdData[scdIdx + k]);
#endif
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


void CustomMatcher::matchV2(const Mat& query, const Mat& train, vector<vector<DMatch>>& matches)
{

    __int64* fstData = (__int64*)query.data;
    __int64* scdData = (__int64*)train.data;

    matches.resize(query.rows, vector<DMatch>(2, DMatch(-1,-1, numeric_limits<float>::max())));

    // TODO(alessandro.pieropan@gmail.com): can be optimized to make it faster
    for (size_t i = 0; i < query.rows; ++i)
    {
        DMatch& best = matches.at(i)[0];
        DMatch& second = matches.at(i)[1];
        best.queryIdx = i;
        second.queryIdx = i;

        //list<DMatch> bestMatches(bestNum, DMatch(i, -1, numeric_limits<float>::max()));

        for (size_t j = 0; j < train.rows; ++j)
        {
            int fstIdx = 8 * i;
            int scdIdx = 8 * j;

            int distance = 0;

            for (size_t k = 0; k < 8; ++k)
            {
#ifdef __arm__
		distance += __builtin_popcountll(fstData[fstIdx + k] ^ scdData[scdIdx + k]);
#else
		distance += _mm_popcnt_u64(fstData[fstIdx + k] ^ scdData[scdIdx + k]);
#endif
            }

            if(distance < best.distance)
            {
                second.distance = best.distance;
                second.trainIdx = best.trainIdx;
                best.distance = distance;
                best.trainIdx = j;
            }
            else if(distance < second.distance)
            {
                second.distance = distance;
                second.trainIdx = j;
            }
        }
    }
}

void CustomMatcher::match32(const cv::Mat& query, const cv::Mat& train, int bestNum,
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
#ifdef __arm__
		distance += __builtin_popcountll(fstData[fstIdx + k] ^ scdData[scdIdx + k]);
#else
		distance += _mm_popcnt_u64(fstData[fstIdx + k] ^ scdData[scdIdx + k]);
#endif
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
