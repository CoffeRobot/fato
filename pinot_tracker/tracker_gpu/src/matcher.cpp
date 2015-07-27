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


#include "Matcher.h"
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
