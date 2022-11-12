//
//  ViewController.m
//  ImageMatch

#import "ViewController.h"
#import <opencv2/imgproc/imgproc_c.h>
#import <opencv2/objdetect/objdetect.hpp>
#import "UIImageCVMatConverter.h"

@interface ViewController ()

@end


@implementation ViewController
@synthesize imageView;

- (void)dealloc {
    
}

#pragma mark -
#pragma mark OpenCV Support Methods

// NOTE you SHOULD cvReleaseImage() for the return value when end of the code.
- (IplImage *)CreateIplImageFromUIImage:(UIImage *)image {
	CGImageRef imageRef = image.CGImage;
    
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	IplImage *iplimage = cvCreateImage(cvSize(image.size.width, image.size.height), IPL_DEPTH_8U, 4);
	CGContextRef contextRef = CGBitmapContextCreate(iplimage->imageData, iplimage->width, iplimage->height,
													iplimage->depth, iplimage->widthStep,
													colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);
	CGContextDrawImage(contextRef, CGRectMake(0, 0, image.size.width, image.size.height), imageRef);
	CGContextRelease(contextRef);
	CGColorSpaceRelease(colorSpace);
    
	IplImage *ret = cvCreateImage(cvGetSize(iplimage), IPL_DEPTH_8U, 3);
	cvCvtColor(iplimage, ret, CV_RGBA2BGR);
	cvReleaseImage(&iplimage);
    
	return ret;
}

- (UIImage *)UIImageFromIplImage:(IplImage *)image {
	NSLog(@"IplImage (%d, %d) %d bits by %d channels, %d bytes/row %s", image->width, image->height, image->depth, image->nChannels, image->widthStep, image->channelSeq);
    
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
	NSData *data = [NSData dataWithBytes:image->imageData length:image->imageSize];
	CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
	CGImageRef imageRef = CGImageCreate(image->width, image->height,
										image->depth, image->depth * image->nChannels, image->widthStep,
                                        colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault,
										provider, NULL, false, kCGRenderingIntentDefault);
	UIImage *ret = [UIImage imageWithCGImage:imageRef];
	CGImageRelease(imageRef);
	CGDataProviderRelease(provider);
	CGColorSpaceRelease(colorSpace);
	return ret;
}

#pragma mark -
#pragma mark Utilities for internal use

- (void)showProgressIndicator:(NSString *)text {
	//[UIApplication sharedApplication].networkActivityIndicatorVisible = YES;
	self.view.userInteractionEnabled = FALSE;
	if(!progressHUD) {
		CGFloat w = 160.0f, h = 120.0f;
		progressHUD = [[UIProgressHUD alloc] initWithFrame:CGRectMake((self.view.frame.size.width-w)/2, (self.view.frame.size.height-h)/2, w, h)];
		[progressHUD setText:text];
		[progressHUD showInView:self.view];
	}
}

- (void)hideProgressIndicator {
	self.view.userInteractionEnabled = TRUE;
	if(progressHUD) {
		[progressHUD hide];
		progressHUD = nil;
	}
}



#pragma mark -
#pragma mark IBAction

- (IBAction)loadImage:(id)sender {
	if(!actionSheetAction) {
		UIActionSheet *actionSheet = [[UIActionSheet alloc] initWithTitle:@""
																 delegate:self cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil
														otherButtonTitles:@"Use Photo from Library", @"Take Photo with Camera", @"Use Default", nil];
		actionSheet.actionSheetStyle = UIActionSheetStyleDefault;
		actionSheetAction = ActionSheetToSelectTypeOfSource;
		[actionSheet showInView:self.view];
	}
}

//- (IBAction)saveImage:(id)sender {
//	if(imageView.image) {
//		[self showProgressIndicator:@"Saving"];
//		UIImageWriteToSavedPhotosAlbum(imageView.image, self, @selector(finishUIImageWriteToSavedPhotosAlbum:didFinishSavingWithError:contextInfo:), nil);
//	}
//}

- (void)finishUIImageWriteToSavedPhotosAlbum:(UIImage *)image didFinishSavingWithError:(NSError *)error contextInfo:(void *)contextInfo {
	[self hideProgressIndicator];
}


#pragma mark -
#pragma mark UIViewControllerDelegate

- (void)viewDidLoad {
	[super viewDidLoad];
    [[UIApplication sharedApplication] setStatusBarHidden:YES withAnimation:UIStatusBarAnimationSlide];

    [self checkAndProvideIOS7Layout];
}

-(IBAction)check:(id)sender
{
    //    UIImage *img=[UIImageCVMatConverter getDetector:imageView.image imageWith:[UIImage imageNamed:@"gopher.png"]];
    
    // int img1 = [UIImageCVMatConverter findObjectSURF:[UIImage imageNamed:@"snakeObj.png"] andScene:imageView.image withHessianValue:50];
    
    
    
    int mojave1 = [UIImageCVMatConverter findObjectSURF:[UIImage imageNamed:@"mojaveObj1.png"] andScene:imageView.image withHessianValue:50];
    int mojave2 = [UIImageCVMatConverter findObjectSURF:[UIImage imageNamed:@"mojaveObj2.png"] andScene:imageView.image withHessianValue:50];
    int mojaveAvgValue=mojave1+mojave2;
    
    
    int gopher1 = [UIImageCVMatConverter findObjectSURF:[UIImage imageNamed:@"gopherObj.png"] andScene:imageView.image withHessianValue:50];
    int wdb1 =[UIImageCVMatConverter findObjectSURF:[UIImage imageNamed:@"wdbobj1.png"] andScene:imageView.image withHessianValue:50];
    int wdb2 =[UIImageCVMatConverter findObjectSURF:[UIImage imageNamed:@"wdbobj2.png"] andScene:imageView.image withHessianValue:50];
    NSString *str1=[NSString stringWithFormat:@"mojave: %@",(mojaveAvgValue==0)?@"No match":@"Match"];
    NSString *str2=[NSString stringWithFormat:@"gopher: %@",(gopher1==0)?@"No match":@"Match"];
    
    NSString *str3=[NSString stringWithFormat:@"wdb: %@",(wdb1 + wdb2==0)?@"No match":@"Match"];
    UIAlertView *av=[[UIAlertView alloc]initWithTitle:@"Result" message:@"" delegate:nil cancelButtonTitle:@"Ok" otherButtonTitles:str1,str2,str3,nil];
    [av show];
}

-(void)viewDidAppear:(BOOL)animated
{
    // [self loadImage:nil];
}
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation {
	return NO;
}

#pragma mark -
#pragma mark UIActionSheetDelegate

- (void)actionSheet:(UIActionSheet *)actionSheet clickedButtonAtIndex:(NSInteger)buttonIndex {
	switch(actionSheetAction) {
		case ActionSheetToSelectTypeOfSource: {
			UIImagePickerControllerSourceType sourceType;
			if (buttonIndex == 0) {
				sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
			} else if(buttonIndex == 1) {
				sourceType = UIImagePickerControllerSourceTypeCamera;
			} else if(buttonIndex == 2) {
			   // NSString *path = [[NSBundle mainBundle] pathForResource:@"mojaveScene1" ofType:@"png"];
               // NSString *path = [[NSBundle mainBundle] pathForResource:@"gopherScene" ofType:@"jpg"];
                NSString *path = [[NSBundle mainBundle] pathForResource:@"wdbscene" ofType:@"png"];
				imageView.image = [UIImage imageWithContentsOfFile:path];
				break;
			} else {
				// Cancel
				break;
			}
			if([UIImagePickerController isSourceTypeAvailable:sourceType])
            {
				UIImagePickerController *picker = [[UIImagePickerController alloc] init];
				picker.sourceType = sourceType;
				picker.delegate = self;
                [self presentViewController:picker animated:YES completion:nil];
			}
			break;
		}
		case ActionSheetToSelectTypeOfMarks: {
			if(buttonIndex != 0 && buttonIndex != 1) {
				break;
			}
            
			UIImage *image = nil;
			if(buttonIndex == 1) {
				NSString *path = [[NSBundle mainBundle] pathForResource:@"laughing_man" ofType:@"png"];
				image = [UIImage imageWithContentsOfFile:path];
			}
            
			[self showProgressIndicator:@"Detecting"];
			break;
		}
	}
	actionSheetAction = 0;
}

#pragma mark -
#pragma mark UIImagePickerControllerDelegate

- (UIImage *)scaleAndRotateImage:(UIImage *)image {
	static int kMaxResolution = 640;
	
	CGImageRef imgRef = image.CGImage;
	CGFloat width = CGImageGetWidth(imgRef);
	CGFloat height = CGImageGetHeight(imgRef);
	
	CGAffineTransform transform = CGAffineTransformIdentity;
	CGRect bounds = CGRectMake(0, 0, width, height);
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
	
	UIGraphicsBeginImageContext(bounds.size);
	CGContextRef context = UIGraphicsGetCurrentContext();
	if (orient == UIImageOrientationRight || orient == UIImageOrientationLeft) {
		CGContextScaleCTM(context, -scaleRatio, scaleRatio);
		CGContextTranslateCTM(context, -height, 0);
	} else {
		CGContextScaleCTM(context, scaleRatio, -scaleRatio);
		CGContextTranslateCTM(context, 0, -height);
	}
	CGContextConcatCTM(context, transform);
	CGContextDrawImage(UIGraphicsGetCurrentContext(), CGRectMake(0, 0, width, height), imgRef);
	UIImage *imageCopy = UIGraphicsGetImageFromCurrentImageContext();
	UIGraphicsEndImageContext();
	
	return imageCopy;
}

- (void)imagePickerController:(UIImagePickerController *)picker
		didFinishPickingImage:(UIImage *)image
				  editingInfo:(NSDictionary *)editingInfo
{
	imageView.image = [self scaleAndRotateImage:image];
	[picker dismissModalViewControllerAnimated:YES];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker
{
    [picker dismissViewControllerAnimated:YES completion:nil];
}
@end
