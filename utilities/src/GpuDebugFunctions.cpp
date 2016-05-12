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

#include "GpuDebugFunctions.h"
#include <random>
#include <iostream>
#include "ToString.h"

using namespace std;
using namespace cv;

namespace fato{

void DrawFlowPoints(const vector<Point2f>* points, const vector<FatoStatus>* pointsStatus,
	const vector<int>* pointsIds, Mat& out)
{
  //cout << "P " << points->size() << " S " << pointsStatus->size() << " I "
  //     << pointsIds->size() << endl;
	for (int i = 0; i < points->size(); ++i)
	{
		const int id = pointsIds->at(i);
    if (pointsStatus->at(id) == FatoStatus::TRACK)
		{
			circle(out, points->at(i), 3, Scalar(255, 0, 0), 1);
		}
    else if(pointsStatus->at(id) == FatoStatus::MATCH)
    {
      circle(out, points->at(i), 3, Scalar(0, 255, 0), 1);
    }
    /*else if (pointsStatus->at(id) == Status::NOCLUSTER)
    {
      circle(out, points->at(i), 3, Scalar(0, 255, 255), 1);
    }
    else if (pointsStatus->at(id) == Status::LOST)
    {
      circle(out, points->at(i), 3, Scalar(255, 255, 255), 1);
    }*/
	}
}

void DrawDetectedPoints(const vector<Point2f>* initPts, const vector<Point2f>* updPts,
	const vector<FatoStatus>* ptsStatus, const vector<int>* ptsIds, Mat&out)
{
	int imgOffset = out.cols / 2;
	
	random_device rd;
	default_random_engine dre(rd());
	uniform_int_distribution<unsigned int> uniform_dist(0, 255);

	for (size_t i = 0; i < updPts->size(); i++)
	{
		const int id = ptsIds->at(i);
		if (ptsStatus->at(i) == FatoStatus::MATCH)
		{
			Scalar color(uniform_dist(dre), uniform_dist(dre), uniform_dist(dre));

			const Point2f& src = initPts->at(id);
			Point2f dst = updPts->at(i);
			dst.x += imgOffset;

			circle(out, src, 3, color, 1);
			circle(out, dst, 3, color, 1);
			line(out, src, dst, color, 1);
			
		}
	}
}

void drawVotesGPU(const vector<Point2f>* points, const vector<FatoStatus>* pointsStatus,
	const vector<Point2f>* votes, const vector<int>* pointsIds, Mat& out)
{
	for (int i = 0; i < points->size(); ++i)
	{
		const int id = pointsIds->at(i);
		
		if (pointsStatus->at(id) == FatoStatus::TRACK)
		{
			const Point2f& point = points->at(i);
			const Point2f& vote = votes->at(i);
			circle(out, vote, 2, Scalar(0, 255, 0), -1);
			line(out, point, vote, Scalar(0, 255, 0), 1);
		}
    /*if (pointsStatus->at(id) == Status::NOCLUSTER)
    {
      const Point2f& point = points->at(i);
      const Point2f& vote = votes->at(i);
      circle(out, vote, 2, Scalar(0, 255, 255), -1);
      line(out, point, vote, Scalar(0, 255, 255), 1);
    }
    if (pointsStatus->at(id) == Status::MATCH)
    {
      const Point2f& point = points->at(i);
      const Point2f& vote = votes->at(i);
      circle(out, vote, 2, Scalar(0, 0, 255), -1);
      line(out, point, vote, Scalar(0, 0, 255), 1);
    }
    /*if (pointsStatus->at(id) == Status::LOST)
    {
      const Point2f& point = points->at(i);
      const Point2f& vote = votes->at(i);
      circle(out, vote, 2, Scalar(0, 0, 255), -1);
      line(out, point, vote, Scalar(0, 0, 255), 1);
    }*/
	}
}

void printPointsStatus(const vector<Point2f>* points, const vector<FatoStatus>* pointsStatus,
	const vector<Point2f>* votes, const vector<int>* pointsIds, ofstream& file)
{
	for (int i = 0; i < points->size(); ++i)
	{
		const int id = pointsIds->at(i);

		file << "P: " << toString(points->at(i)) << " V: " << toString(votes->at(i))
			 << "S: " << toString(pointsStatus->at(id)) << "\n";
	}
}

} // end namesapce
