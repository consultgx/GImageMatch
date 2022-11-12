//
//  UIImageCVMatConverter.m

#import "UIImageCVMatConverter.h"
#import "TemplateMatch.h"
#import <opencv2/objdetect/objdetect.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"



@implementation UIImageCVMatConverter

+(UIImage *)getDetector:(UIImage *)sceneImage imageWith:(UIImage *)objectImage
{
    cv::Mat sceneImageMat, objectImageMat1;
    cv::vector<cv::KeyPoint> sceneKeypoints, objectKeypoints1;
    cv::Mat sceneDescriptors, objectDescriptors1;
    cv::StarDetector *surfDetector;
    cv::FlannBasedMatcher flannMatcher;
    cv::vector<cv::DMatch> matches;
    int minHessian;
    double minDistMultiplier;
    
    minHessian = 50;
    minDistMultiplier= 3;
    surfDetector = new cv::StarFeatureDetector(minHessian);
    
    sceneImageMat = cv::Mat(sceneImage.size.height, sceneImage.size.width, CV_8UC1);
    objectImageMat1 = cv::Mat(objectImage.size.height, objectImage.size.width, CV_8UC1);
    
    cv::cvtColor([self cvMatFromUIImage:sceneImage], sceneImageMat, CV_RGB2GRAY);
    cv::cvtColor([self cvMatFromUIImage:objectImage], objectImageMat1, CV_RGB2GRAY);
    
    if (!sceneImageMat.data || !objectImageMat1.data) {
        NSLog(@"NO DATA");
    }
    
    surfDetector->detect(sceneImageMat, sceneKeypoints);
    surfDetector->detect(objectImageMat1, objectKeypoints1);
    
    cv::SurfDescriptorExtractor extractor;
    
   
    
    extractor.compute(objectImageMat1, objectKeypoints1, objectDescriptors1 );
    extractor.compute(sceneImageMat,sceneKeypoints, sceneDescriptors );
    
    
    flannMatcher.match(objectDescriptors1, sceneDescriptors, matches);
    
    double max_dist = 0; double min_dist = 100;
    
    for( int i = 0; i < objectDescriptors1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    cv::vector<cv::DMatch> goodMatches;
    for( int i = 0; i < objectDescriptors1.rows; i++ )
    {
        if( matches[i].distance < minDistMultiplier*min_dist )
        {
            goodMatches.push_back( matches[i]);
        }
    }
    NSLog(@"Good matches found: %lu", goodMatches.size());
    
    cv::Mat imageMatches;
    cv::drawMatches(objectImageMat1, objectKeypoints1, sceneImageMat, sceneKeypoints, goodMatches, imageMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    cv::vector<cv::Point2f> obj;
    cv::vector<cv::Point2f> scene;
    for( int i = 0; i < goodMatches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( objectKeypoints1[ goodMatches[i].queryIdx ].pt );
        scene.push_back( objectKeypoints1[ goodMatches[i].trainIdx ].pt );
    }
    
    cv::vector<uchar> outputMask;
    cv::Mat hg = cv::findHomography(obj,scene, CV_RANSAC);
    int inlierCounter = 0;
    for (int i = 0; i < outputMask.size(); i++) {
        if (outputMask[i] == 1) {
            inlierCounter++;
        }
    }
    NSLog(@"Inliers percentage: %d", (int)(((float)inlierCounter / (float)outputMask.size()) * 100));
    
    cv::vector<cv::Point2f> objCorners(4);
    objCorners[0] = cv::Point(0,0);
    objCorners[1] = cv::Point( objectImageMat1.cols, 0 );
    objCorners[2] = cv::Point( objectImageMat1.cols, objectImageMat1.rows );
    objCorners[3] = cv::Point( 0, objectImageMat1.rows );
    
    cv::vector<cv::Point2f> scnCorners(4);
    
    cv::perspectiveTransform(objCorners, scnCorners, hg);
    
    cv::line( imageMatches, scnCorners[0] + cv::Point2f( objectImageMat1.cols, 0), scnCorners[1] + cv::Point2f( objectImageMat1.cols, 0), cv::Scalar(0, 255, 0), 4);
    cv::line( imageMatches, scnCorners[1] + cv::Point2f( objectImageMat1.cols, 0), scnCorners[2] + cv::Point2f( objectImageMat1.cols, 0), cv::Scalar( 0, 255, 0), 4);
    cv::line( imageMatches, scnCorners[2] + cv::Point2f( objectImageMat1.cols, 0), scnCorners[3] + cv::Point2f( objectImageMat1.cols, 0), cv::Scalar( 0, 255, 0), 4);
    cv::line( imageMatches, scnCorners[3] + cv::Point2f( objectImageMat1.cols, 0), scnCorners[0] + cv::Point2f( objectImageMat1.cols, 0), cv::Scalar( 0, 255, 0), 4);
    
  return [self UIImageFromCVMat:imageMatches];
}

+(int)findObjectSURF:(UIImage *)objectImg andScene:(UIImage *)sceneImg withHessianValue:(int)hessianValue
{
    cv::Mat objectMat=[self cvMatFromUIImage:objectImg];
   cv::Mat sceneMat=[self cvMatFromUIImage:sceneImg];
//    cv::Mat objectMat=[self cvMatGrayFromUIImage:objectImg];
//   cv::Mat sceneMat=[self cvMatGrayFromUIImage:sceneImg];
    bool objectFound = false;
    float nndrRatio = 0.7f;
    //vector of keypoints
    cv::vector< cv::KeyPoint > keypointsO;
    cv::vector< cv::KeyPoint > keypointsS;
    
    cv::Mat descriptors_object, descriptors_scene;
    
    //-- Step 1: Extract keypoints
    cv::StarFeatureDetector surf(hessianValue);
    surf.detect(sceneMat,keypointsS);
    if(keypointsS.size() < 7)
    {
      return 0;  //Not enough keypoints, object not found
    }
    surf.detect(objectMat,keypointsO);
    if(keypointsO.size() < 7)
    {
        return 0;  //Not enough keypoints, object not found
    }
    
    //-- Step 2: Calculate descriptors (feature vectors)
   cv::SurfDescriptorExtractor extractor;
    extractor.compute( sceneMat, keypointsS, descriptors_scene );
    extractor.compute( objectMat, keypointsO, descriptors_object );
    
    //-- Step 3: Matching descriptor vectors using FLANN matcher
   cv::FlannBasedMatcher matcher;
    descriptors_scene.size(), keypointsO.size(), keypointsS.size();
    std::vector<cv::vector<cv::DMatch >> matches;
    matcher.knnMatch( descriptors_object, descriptors_scene, matches, 2 );
    cv::vector<cv::DMatch > good_matches;
    good_matches.reserve(matches.size());
    
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (matches[i].size() < 2)
            continue;
        
        const cv::DMatch &m1 = matches[i][0];
        const cv::DMatch &m2 = matches[i][1];
        
        if(m1.distance <= nndrRatio * m2.distance)
            good_matches.push_back(m1);
    }
    
    
    cv::Mat outImg;
    if( (good_matches.size() >=7))
    {
        NSLog(@"OBJECT FOUND!");
        
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f > scene;
        
        for( unsigned int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypointsO[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypointsS[ good_matches[i].trainIdx ].pt );
        }
        
       cv::Mat H = findHomography( obj, scene, CV_RANSAC );
        
        
        
        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<cv::Point2f > obj_corners(4);
        obj_corners[0] = cvPoint(0,0);
        obj_corners[1] = cvPoint( objectMat.cols, 0 );
        obj_corners[2] = cvPoint( objectMat.cols, objectMat.rows );
        obj_corners[3] = cvPoint( 0, objectMat.rows );
        
        
        std::vector<cv::Point2f > scene_corners(4);
        
        perspectiveTransform( obj_corners, scene_corners, H);
        
        
        
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        cv::line( outImg, scene_corners[0] , scene_corners[1], cv::Scalar(0, 255, 0), 2 ); //TOP line
        cv::line( outImg, scene_corners[1] , scene_corners[2], cv::Scalar(0, 255, 0), 2 );
        cv::line( outImg, scene_corners[2] , scene_corners[3], cv::Scalar(0, 255, 0), 2 );
        cv::line( outImg, scene_corners[3] , scene_corners[0] , cv::Scalar(0, 255, 0), 2 );
        objectFound=true;
    }
    else {
        NSLog(@"OBJECT NOT FOUND!");
    }
    NSLog(@"Matches found: %ld",matches.size());
     NSLog(@"Good matches found: %ld",good_matches.size());
    return 100*good_matches.size()/matches.size();
}


+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat withUIImage:(UIImage*)image;
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace( image.CGImage );
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    CGFloat widthStep = image.size.width;
    CGContextRef contextRef = CGBitmapContextCreate( NULL, cols, rows, 8, widthStep*4, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault );
    CGContextDrawImage( contextRef, CGRectMake(0, 0, cols, rows), image.CGImage );
    CGContextSetRGBStrokeColor( contextRef, 1, 0, 0, 1 );
    CGImageRef cgImage = CGBitmapContextCreateImage( contextRef );
    UIImage* result = [UIImage imageWithCGImage:cgImage];
    CGImageRelease( cgImage );
    CGContextRelease( contextRef );
    CGColorSpaceRelease( colorSpace );
    return result;
}

+ (UIImage*) templateMatchImage:(UIImage *)image
                          patch:(UIImage *)patch
                         method:(int)method
{
    cv::Mat imageMat = [self cvMatFromUIImage:image];
    cv::Mat patchMat = [self cvMatFromUIImage:patch];
    
    cv::Mat matchImage = TemplateMatch::matchImage(imageMat,patchMat,method);
    
    UIImage* result = [self UIImageFromCVMat:matchImage];
    return result;
}

+(CGPoint)getlocationPoint
{
   cv::Point matLocation=TemplateMatch::getMatchLoc();
    
    CGPoint tempPt;
    tempPt.x=matLocation.x;
    tempPt.y=matLocation.y;
    return tempPt;
}

//+(CGPoint)findCord:()

+(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    if ( cvMat.elemSize() == 1 ) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    }
    else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData( (__bridge CFDataRef)data );
    CGImageRef imageRef = CGImageCreate( cvMat.cols, cvMat.rows, 8, 8 * cvMat.elemSize(), cvMat.step[0], colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault );
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease( imageRef );
    CGDataProviderRelease( provider );
    CGColorSpaceRelease( colorSpace );
    return finalImage;
}
+ (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace( image.CGImage );
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    cv::Mat cvMat( rows, cols, CV_8UC4 );
    CGContextRef contextRef = CGBitmapContextCreate( cvMat.data, cols, rows, 8, cvMat.step[0], colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault );
    CGContextDrawImage( contextRef, CGRectMake(0, 0, cols, rows), image.CGImage );
    CGContextRelease( contextRef );
    CGColorSpaceRelease( colorSpace );
    return cvMat;
}

+ (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image
{
    cv::Mat cvMat = [UIImageCVMatConverter cvMatFromUIImage:image];
    cv::Mat grayMat;
    if ( cvMat.channels() == 1 ) {
        grayMat = cvMat;
    }
    else {
        grayMat = cv :: Mat( cvMat.rows,cvMat.cols, CV_8UC1 );
        cv::cvtColor( cvMat, grayMat, CV_BGR2GRAY );
    }
    return grayMat;
}

+ (UIImage *)scaleAndRotateImageBackCamera:(UIImage *)image
{
    static int kMaxResolution = 640;
    CGImageRef imgRef = image.CGImage;
    CGFloat width = CGImageGetWidth( imgRef );
    CGFloat height = CGImageGetHeight( imgRef );
    CGAffineTransform transform = CGAffineTransformIdentity;
    CGRect bounds = CGRectMake( 0, 0, width, height );
    if ( width > kMaxResolution || height > kMaxResolution ) {
        CGFloat ratio = width/height;
        if ( ratio > 1 ) {
            bounds.size.width = kMaxResolution;
            bounds.size.height = bounds.size.width / ratio;
        }
        else {
            bounds.size.height = kMaxResolution;
            bounds.size.width = bounds.size.height * ratio;
        }
    }
    CGFloat scaleRatio = bounds.size.width / width;
    CGSize imageSize = CGSizeMake( CGImageGetWidth(imgRef), CGImageGetHeight(imgRef) );
    CGFloat boundHeight;
    UIImageOrientation orient = image.imageOrientation;
    switch( orient ) {
        case UIImageOrientationUp:
            transform = CGAffineTransformIdentity;
            break;
        case UIImageOrientationUpMirrored:
            transform = CGAffineTransformMakeTranslation(imageSize.width, 0.0);
            transform = CGAffineTransformScale(transform, -1.0, 1.0);
            break;
        case UIImageOrientationDown:
            transform = CGAffineTransformMakeTranslation(imageSize.width, imageSize.height);
            transform = CGAffineTransformRotate(transform, M_PI);
            break;
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformMakeTranslation(0.0, imageSize.height);
            transform = CGAffineTransformScale(transform, 1.0, -1.0);
            break;
        case UIImageOrientationLeftMirrored:
            boundHeight = bounds.size.height;
            bounds.size.height = bounds.size.width;
            bounds.size.width = boundHeight;
            transform = CGAffineTransformMakeTranslation(imageSize.height, imageSize.width);
            transform = CGAffineTransformScale(transform, -1.0, 1.0);
            transform = CGAffineTransformRotate(transform, 3.0 * M_PI / 2.0);
            break;
        case UIImageOrientationLeft:
            boundHeight = bounds.size.height;
            bounds.size.height = bounds.size.width;
            bounds.size.width = boundHeight;
            transform = CGAffineTransformMakeTranslation(0.0, imageSize.width);
            transform = CGAffineTransformRotate(transform, 3.0 * M_PI / 2.0);
            break;
        case UIImageOrientationRightMirrored:
            boundHeight = bounds.size.height;
            bounds.size.height = bounds.size.width;
            bounds.size.width = boundHeight;
            transform = CGAffineTransformMakeScale(-1.0, 1.0);
            transform = CGAffineTransformRotate(transform, M_PI / 2.0);
            break;
        case UIImageOrientationRight:
            boundHeight = bounds.size.height;
            bounds.size.height = bounds.size.width;
            bounds.size.width = boundHeight;
            transform = CGAffineTransformMakeTranslation(imageSize.height, 0.0);
            transform = CGAffineTransformRotate(transform, M_PI / 2.0);
            break;
        default:
            [NSException raise:NSInternalInconsistencyException format:@"Invalid image orientation"];
    }
    UIGraphicsBeginImageContext( bounds.size );
    CGContextRef context = UIGraphicsGetCurrentContext();
    if ( orient == UIImageOrientationRight || orient == UIImageOrientationLeft ) {
        CGContextScaleCTM( context, -scaleRatio, scaleRatio );
        CGContextTranslateCTM( context, -height, 0 );
    }
    else {
        CGContextScaleCTM( context, scaleRatio, -scaleRatio );
        CGContextTranslateCTM( context, 0, -height );
    }
    CGContextConcatCTM( context, transform );
    CGContextDrawImage( UIGraphicsGetCurrentContext(), CGRectMake(0, 0, width, height), imgRef );
    UIImage *returnImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return returnImage;
}

+ (UIImage *)scaleAndRotateImageFrontCamera:(UIImage *)image
{
    static int kMaxResolution = 640;
    CGImageRef imgRef = image.CGImage;
    CGFloat width = CGImageGetWidth(imgRef);
    CGFloat height = CGImageGetHeight(imgRef);
    CGAffineTransform transform = CGAffineTransformIdentity;
    CGRect bounds = CGRectMake( 0, 0, width, height);
    if (width > kMaxResolution || height > kMaxResolution) {
        CGFloat ratio = width/height;
        if (ratio > 1) {
            bounds.size.width = kMaxResolution;
            bounds.size.height = bounds.size.width / ratio;
        } else {
            bounds.size.height = kMaxResolution;
            bounds.size.width = bounds.size.height * ratio;
        }
    }
    
    CGFloat scaleRatio = bounds.size.width / width;
    CGSize imageSize = CGSizeMake(CGImageGetWidth(imgRef), CGImageGetHeight(imgRef));
    CGFloat boundHeight;
    UIImageOrientation orient = image.imageOrientation;
    switch(orient) {
        case UIImageOrientationUp:
            transform = CGAffineTransformIdentity;
            break;
        case UIImageOrientationUpMirrored:
            transform = CGAffineTransformMakeTranslation(imageSize.width, 0.0);
            transform = CGAffineTransformScale(transform, -1.0, 1.0);
            break;
        case UIImageOrientationDown:
            transform = CGAffineTransformMakeTranslation(imageSize.width, imageSize.height);
            transform = CGAffineTransformRotate(transform, M_PI);
            break;
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformMakeTranslation(0.0, imageSize.height);
            transform = CGAffineTransformScale(transform, 1.0, -1.0);
            break;
        case UIImageOrientationLeftMirrored:
            boundHeight = bounds.size.height;
            bounds.size.height = bounds.size.width;
            bounds.size.width = boundHeight;
            transform = CGAffineTransformMakeTranslation(imageSize.height, imageSize.width);
            transform = CGAffineTransformScale(transform, -1.0, 1.0);
            transform = CGAffineTransformRotate(transform, 3.0 * M_PI / 2.0);
            break;
        case UIImageOrientationLeft:
            boundHeight = bounds.size.height;
            bounds.size.height = bounds.size.width;
            bounds.size.width = boundHeight;
            transform = CGAffineTransformMakeTranslation(0.0, imageSize.width);
            transform = CGAffineTransformRotate(transform, 3.0 * M_PI / 2.0);
            break;
        case UIImageOrientationRight:
        case UIImageOrientationRightMirrored:
            boundHeight = bounds.size.height;
            bounds.size.height = bounds.size.width;
            bounds.size.width = boundHeight;
            transform = CGAffineTransformMakeScale(-1.0, 1.0);
            transform = CGAffineTransformRotate(transform, M_PI / 2.0);
            break;
        default:
            [NSException raise:NSInternalInconsistencyException format:@"Invalid image orientation"];
    }
    UIGraphicsBeginImageContext( bounds.size );
    CGContextRef context = UIGraphicsGetCurrentContext();
    if ( orient == UIImageOrientationRight || orient == UIImageOrientationLeft ) {
        CGContextScaleCTM(context, -scaleRatio, scaleRatio);
        CGContextTranslateCTM(context, -height, 0);
    }
    else {
        CGContextScaleCTM(context, scaleRatio, -scaleRatio);
        CGContextTranslateCTM(context, 0, -height);
    }
    CGContextConcatCTM( context, transform );
    CGContextDrawImage( UIGraphicsGetCurrentContext(), CGRectMake(0, 0, width, height), imgRef );
    UIImage *returnImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return returnImage;
}

// NOTE you SHOULD cvReleaseImage() for the return value when end of the code.
+ (IplImage *)CreateIplImageFromUIImage:(UIImage *)image {
    // Getting CGImage from UIImage
    CGImageRef imageRef = image.CGImage;
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    // Creating temporal IplImage for drawing
    IplImage *iplimage = cvCreateImage(
                                       cvSize(image.size.width,image.size.height), IPL_DEPTH_8U, 4
                                       );
    // Creating CGContext for temporal IplImage
    CGContextRef contextRef = CGBitmapContextCreate(
                                                    iplimage->imageData, iplimage->width, iplimage->height,
                                                    iplimage->depth, iplimage->widthStep,
                                                    colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault
                                                    );
    // Drawing CGImage to CGContext
    CGContextDrawImage(
                       contextRef,
                       CGRectMake(0, 0, image.size.width, image.size.height),
                       imageRef
                       );
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    // Creating result IplImage
    IplImage *ret = cvCreateImage(cvGetSize(iplimage), IPL_DEPTH_8U, 3);
    cvCvtColor(iplimage, ret, CV_RGBA2BGR);
    cvReleaseImage(&iplimage);
    
    return ret;
}

// NOTE You should convert color mode as RGB before passing to this function
+ (UIImage *)UIImageFromIplImage:(IplImage *)image {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    // Allocating the buffer for CGImage
    NSData *data =
    [NSData dataWithBytes:image->imageData length:image->imageSize];
    CGDataProviderRef provider =
    CGDataProviderCreateWithCFData((CFDataRef)data);
    // Creating CGImage from chunk of IplImage
    CGImageRef imageRef = CGImageCreate(
                                        image->width, image->height,
                                        image->depth, image->depth * image->nChannels, image->widthStep,
                                        colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault,
                                        provider, NULL, false, kCGRenderingIntentDefault
                                        );
    // Getting UIImage from CGImage
    UIImage *ret = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return ret;
}

@end
