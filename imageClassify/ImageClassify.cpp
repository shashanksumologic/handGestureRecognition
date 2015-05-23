#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include "ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#endif

#include "preprocessing.h"
#include "ImageClassify.h"


void ImageClassify::getData()
{
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	int i,j;
	for(i =0; i<classes; i++){
		for( j = 1; j<= train_samples; j++){
			
			//Load file
			if(j<10)
				sprintf(file,"%s%d/%d0%d.pbm",file_path, i, 0 , j);
			else
				sprintf(file,"%s%d/%d%d.pbm",file_path, i, 0, j);
			src_image = cvLoadImage(file,0);
			if(!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			
			//Set class label
			cvGetRow(trainClasses, &row, i*train_samples + j-1);
			cvSet(&row, cvRealScalar(i));
			//Set data 
			cvGetRow(trainData, &row, i*train_samples + j-1);

			IplImage* img = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
			//convert 8 bits image to 32 float image
			cvConvertScale(&prs_image, img, 0.0039215, 0);

			cvGetSubRect(img, &data, cvRect(0,0, size,size));
			
			CvMat row_header, *row1;
			//convert data matrix sizexsize to vecor
			row1 = cvReshape( &data, &row_header, 0, 1 );
			cvCopy(row1, &row, NULL);
		}
	}
}

void ImageClassify::train()
{
	knn=new CvKNearest( trainData, trainClasses, 0, false, K );
}

float ImageClassify::classify(IplImage* img, int showResult)
{
	IplImage prs_image;
	CvMat data;
	CvMat* nearest=cvCreateMat(1,K,CV_32FC1);
	float result;
	//process file
	prs_image = preprocessing(img, size, size);
	
	//Set data 
	IplImage* img32 = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
	cvConvertScale(&prs_image, img32, 0.0039215, 0);
	cvGetSubRect(img32, &data, cvRect(0,0, size,size));
	CvMat row_header, *row1;
	row1 = cvReshape( &data, &row_header, 0, 1 );

	result=knn->find_nearest(row1,K,0,0,nearest,0);
	
	int accuracy=0;
	for(int i=0;i<K;i++){
		if( nearest->data.fl[i] == result)
                    accuracy++;
	}
	float pre=100*((float)accuracy/(float)K);
	if(showResult==1){
		printf("|\t%d\t| \t%.2f%%  \t| \t%d of %d \t| \n",(int)result,pre,accuracy,K);
		printf(" ---------------------------------------------------------------\n");
	}

	return result;

}

void ImageClassify::test(){
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	int i,j;
	int error=0;
	int testCount=0;
	for(i =0; i<classes; i++){
		for( j = 76; j<= 25+train_samples; j++){
			if(j<100)
			sprintf(file,"%s%d/%d%d.pbm",file_path, i, 0 , j);
                        else
 			sprintf(file,"%s%d/%d.pbm",file_path, i, j);
			src_image = cvLoadImage(file,0);
			if(!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			float r=classify(&prs_image,0);
			if((int)r!=i)
				error++;
			
			testCount++;
		}
	}
	float totalerror=100*(float)error/(float)testCount;
	printf("System Error: %.2f%%\n", totalerror);
	
}

ImageClassify::ImageClassify()
{

	//initial
	sprintf(file_path , "../images/");
	train_samples = 75;
	classes= 10;
	size=40;

	trainData = cvCreateMat(train_samples*classes, size*size, CV_32FC1);
	trainClasses = cvCreateMat(train_samples*classes, 1, CV_32FC1);

	//Get data (get images and process it)
	getData();
	
	//train	
	train();
	//Test	
	test();
	

	
}


