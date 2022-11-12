//
//  Utility.h
//  ImageMatch
//

#import <Foundation/Foundation.h>


#define APPDELEGATE   ((AppDelegate *)[[UIApplication sharedApplication] delegate])

#define isiPhone5 ([[UIScreen mainScreen] bounds].size.height > 480 ? YES :NO)

@interface Utility : NSObject
@end

@interface UIViewController(IOS7)

- (void) checkAndProvideIOS7Layout;
@end


