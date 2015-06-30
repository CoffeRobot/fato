#ifndef UTILITIES_H
#define UTILITIES_H

#include<cmath>

inline float roundUpToNearest(float src_number, float round_to)
{
  if(round_to == 0)
    return src_number;
  else if(src_number > 0)
    return std::ceil(src_number / round_to) * round_to;
  else
    return std::floor(src_number / round_to) * round_to;
}

inline float roundDownToNearest(float src_number, float round_to)
{
  if(round_to == 0)
    return src_number;
  else if(src_number > 0)
    return std::floor(src_number / round_to) * round_to;
  else
    return std::ceil(src_number / round_to) * round_to;
}

inline float getDistance(const cv::Point2f& a, const cv::Point2f& b)
{
  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}



#endif // UTILITIES_H
