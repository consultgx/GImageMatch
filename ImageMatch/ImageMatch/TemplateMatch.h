//
//  TemplateMatch.h
//  ImageMatch
//

#ifndef __ImageMatch__TemplateMatch__
#define __ImageMatch__TemplateMatch__

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
class TemplateMatch
{
public:
    static cv::Mat matchImage (cv::Mat imageMat,
                               cv::Mat patchMat,
                               int method);
   static cv::Point getMatchLoc();
};

#endif /* defined(__ImageMatch__TemplateMatch__) */




