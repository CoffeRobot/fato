#include "ToString.h"

#include <iomanip>

string toString(const Point2f& p)
{
	stringstream ss;
	ss << fixed << setprecision(2) <<"[" << p.x << "," << p.y << "] ";
	return ss.str();
}


string toString(const Status& s)
{
	switch (s)
	{

	case Status::BACKGROUND:
		return "BACKGROUND";
	case Status::INIT:
		return "INIT";
	case Status::MATCH:
		return "MATCH";
	case Status::NOMATCH:
		return "NOMATCH";
	case Status::NOCLUSTER:
		return "NOCLUSTER";
	case Status::TRACK:
		return "TRACK";
	default:
		return "LOST";
	}
}

string toString(const Point3f& point)
{
	stringstream ss;

	ss << "[" << point.x << "," << point.y << "," << point.z << "] ";

	return ss.str();
}

string toString(const Vec3f& point)
{
	stringstream ss;

	ss << "[" << point[0] << "," << point[1] << "," << point[2] << "] ";

	return ss.str();
}