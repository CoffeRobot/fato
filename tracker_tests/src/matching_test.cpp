#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../../tracker/include/feature_matcher.hpp"
#include "../../tracker/include/matcher.h"

#include <iostream>
#include <chrono>


using namespace cv;
using namespace std;
using namespace fato;
using namespace std::chrono;

typedef high_resolution_clock my_clock;

int main(int argc, char** argv)
{


    Mat fst  = imread("../../../src/data/imgs/img1.pgm",0);
    Mat scd  = imread("../../../src/data/imgs/img2.pgm",0);

    BriskMatcher custom;
    Mat fst_dsc, scd_dsc;
    vector<KeyPoint> fst_kps, scd_kps;
    custom.extract(fst, fst_kps, fst_dsc);
    custom.extract(scd, scd_kps, scd_dsc);

    std::vector<std::vector<cv::DMatch>> custom_matches, cv_matches;

    CustomMatcher matcher_custom;
    auto begin_custom = my_clock::now();
    matcher_custom.matchV2(fst_dsc, scd_dsc, custom_matches);
    auto end_custom = my_clock::now();


    cv::BFMatcher matcher(NORM_HAMMING);
    auto begin_cv = my_clock::now();
    matcher.knnMatch(fst_dsc, scd_dsc, cv_matches, 2);
    auto end_cv = my_clock::now();

    for(auto i = 0; i < custom_matches.size(); ++i)
    {
        DMatch mine = custom_matches.at(i)[0];
        DMatch alt = cv_matches.at(i)[0];

        cout << mine.trainIdx << " " << mine.queryIdx << " " << mine.distance << endl;
        cout << alt.trainIdx << " " << alt.queryIdx << " " << alt.distance << endl;
        cout << "\n";
    }

    float ms_custom = duration_cast<nanoseconds>(end_custom - begin_custom).count();
    float ms_cv = duration_cast<nanoseconds>(end_cv - begin_cv).count();

    cout <<fixed << setprecision(3) << "opencv " << ms_cv/1000.0 << " mine " << ms_custom/1000.0 << endl;

    return 0;
}
