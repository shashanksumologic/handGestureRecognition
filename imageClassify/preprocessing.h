#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <ctype.h>
#endif

IplImage preprocessing(IplImage* imgSrc,int new_width, int new_height);
