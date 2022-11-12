//
//  TemplateMatch.cpp
//  ImageMatch
//

#include "TemplateMatch.h"

//[1] #include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/// Global Variables
Mat img; Mat templ; Mat result;
//[1] char* image_window = "Source Image";
//[1] char* result_window = "Result window";
Point matchLoc;
int match_method;
//[1]  int max_Trackbar = 5;

/// Function Headers
Mat MatchingMethod( int, void* );  //[3] (added return value to function)

// [2] /** @function main */
// [2] int main( int argc, char** argv )

Point TemplateMatch::getMatchLoc()
{
    return matchLoc;
}





Mat TemplateMatch::matchImage (Mat image,Mat patch, int method)
// [2]
{
    /// Load image and template
    //[2]  img = imread( argv[1], 1 );
    //[2] templ = imread( argv[2], 1 );
    
    img = image;           //[2]
    templ = patch;         //[2]
    match_method = method; //[2]
    
    /// Create windows
    //[1] namedWindow( image_window, CV_WINDOW_AUTOSIZE );
    //[1] namedWindow( result_window, CV_WINDOW_AUTOSIZE );
    
    /// Create Trackbar
    //[1] char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
    //[1] createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );
    
    Mat result = MatchingMethod( 0, 0 );
    
    //[1] waitKey(0);
    //[2] return 0;
    return result;  //[2]
}


//[3] void MatchingMethod( int, void* )
Mat MatchingMethod( int, void* )

{
    /// Source image to display
    Mat img_display;
    img.copyTo( img_display );
    
    /// Create the result matrix
    int result_cols =  img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    
    result.create( result_cols, result_rows, CV_32FC1 );
    
    /// Do the Matching and Normalize
    matchTemplate( img, templ, result, match_method );
    normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
    
    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal; Point minLoc; Point maxLoc;
   
    
    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
    
    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    {
        matchLoc = minLoc; }
    else
    {
        matchLoc = maxLoc;
    }
    
    /// Show me what you got
    rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
    rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
    
    //[1]  imshow( image_window, img_display );
    //[1] imshow( result_window, result );
    
    return img_display; //[3] add return value
}
