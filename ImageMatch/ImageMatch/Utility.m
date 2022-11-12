//
//  Utility.m
//  ImageMatch

#import "Utility.h"


@implementation Utility

@end

@implementation UIViewController(IOS7)

- (void) checkAndProvideIOS7Layout {
    if ([self respondsToSelector:@selector(setEdgesForExtendedLayout:)]) {
        [self setEdgesForExtendedLayout:UIRectEdgeNone];
    }
}

@end
