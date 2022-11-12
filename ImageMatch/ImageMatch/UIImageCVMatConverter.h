//
//  UIImageCVMatConverter.h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include "opencv2/imgproc/imgproc.hpp"

@interface UIImageCVMatConverter : NSObject
{
}
//+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat;
//+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat withUIImage:(UIImage*)image;
//+ (cv::Mat)cvMatFromUIImage:(UIImage *)image;
//+ (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image;
+ (UIImage *)scaleAndRotateImageFrontCamera:(UIImage *)image;
+ (UIImage *)scaleAndRotateImageBackCamera:(UIImage *)image;
+ (IplImage *)CreateIplImageFromUIImage:(UIImage *)image;
+ (UIImage *)UIImageFromIplImage:(IplImage *)image;
+ (UIImage *) templateMatchImage:(UIImage*)image
                          patch:(UIImage*)patch
                         method:(int)method;
+ (CGPoint)getlocationPoint;
+(UIImage *)getDetector:(UIImage *)sceneImage imageWith:(UIImage *)objectImage;
+(int)findObjectSURF:(UIImage *)objectImg andScene:(UIImage *)sceneImg withHessianValue:(int)hessianValue;
@end
