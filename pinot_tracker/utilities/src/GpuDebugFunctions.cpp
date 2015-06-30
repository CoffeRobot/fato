#include "GpuDebugFunctions.h"
#include <random>
#include <iostream>
#include "ToString.h"

using namespace std;

void DrawFlowPoints(const vector<Point2f>* points, const vector<Status>* pointsStatus,
	const vector<int>* pointsIds, Mat& out)
{
  //cout << "P " << points->size() << " S " << pointsStatus->size() << " I "
  //     << pointsIds->size() << endl;
	for (int i = 0; i < points->size(); ++i)
	{
		const int id = pointsIds->at(i);
    if (pointsStatus->at(id) == Status::TRACK)
		{
			circle(out, points->at(i), 3, Scalar(255, 0, 0), 1);
		}
    else if(pointsStatus->at(id) == Status::MATCH)
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
	const vector<Status>* ptsStatus, const vector<int>* ptsIds, Mat&out)
{
	int imgOffset = out.cols / 2;
	
	random_device rd;
	default_random_engine dre(rd());
	uniform_int_distribution<unsigned int> uniform_dist(0, 255);

	for (size_t i = 0; i < updPts->size(); i++)
	{
		const int id = ptsIds->at(i);
		if (ptsStatus->at(i) == Status::MATCH)
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

void drawVotesGPU(const vector<Point2f>* points, const vector<Status>* pointsStatus,
	const vector<Point2f>* votes, const vector<int>* pointsIds, Mat& out)
{
	for (int i = 0; i < points->size(); ++i)
	{
		const int id = pointsIds->at(i);
		
		if (pointsStatus->at(id) == Status::TRACK)
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

void printPointsStatus(const vector<Point2f>* points, const vector<Status>* pointsStatus,
	const vector<Point2f>* votes, const vector<int>* pointsIds, ofstream& file)
{
	for (int i = 0; i < points->size(); ++i)
	{
		const int id = pointsIds->at(i);

		file << "P: " << toString(points->at(i)) << " V: " << toString(votes->at(i))
			 << "S: " << toString(pointsStatus->at(id)) << "\n";
	}
}
