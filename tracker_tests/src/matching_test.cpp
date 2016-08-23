#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../../tracker/include/feature_matcher.hpp"

using namespace cv;
using namespace std;
using namespace fato;

int main(int argc, char** argv)
{


    Mat fst  = imread("",0);
    Mat scd  = imread("",0);

    BriskMatcher matcher;
    Mat fst_dsc, scd_dsc;
    vector<KeyPoint> fst_kps, scd_kps;
    matcher.extract(fst, fst_kps, fst_dsc);
    matcher.extract(scd, scd_kps, scd_dsc);

    return 0;
}