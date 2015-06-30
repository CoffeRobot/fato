#ifndef TOSTRING_H
#define TOSTRING_H

#include <string>
#include <opencv2/core/core.hpp>
#include <sstream>


#include "Constants.h"

using namespace std;
using namespace cv;


string toString(const Point2f& p);

string toString(const Status& s);

string toString(const Point3f& point);

string toString(const Vec3f& point);


#endif