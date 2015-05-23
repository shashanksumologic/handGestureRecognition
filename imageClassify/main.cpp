#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ImageClassify.h"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
int c=0;
int flag=0;
int flagc=0;
/** Function Headers */
void detectAndDisplay( Mat frame );
Mat imline;
int prevX = -1, prevY = -1;

/** Global variables */
String face_cascade_name = "fist.xml";
CascadeClassifier face_cascade;
std::vector<Rect> faces;
Mat diffImage(Mat t0,Mat t1,Mat t2)
{
  Mat d1,d2,motion;
  absdiff(t2, t1, d1);
  absdiff(t1, t0, d2);
  bitwise_and(d1, d2, motion);
  return motion;
}
/** @function main */
int main( void )
{
	VideoCapture capture;
	Mat p_frame,frame,n_frame;
	Mat imgThresholded;
        ImageClassify imageClassify;
	Size f(128,128);
	int iLowH = 0;
	int iHighH = 40;

	int iLowS = 30; 
	int iHighS = 150;

	int iLowV = 60;
	int iHighV = 255;
	namedWindow("Control", CV_WINDOW_AUTOSIZE); 

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 255); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 255);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

	//-- 1. Load the cascades
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
	//if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

	//-- 2. Read the video stream
	capture.open( -1 );
	if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
	capture.read(p_frame);
        capture.read(frame);
	capture.read(n_frame);
 	imline= Mat::zeros( frame.size(), CV_8UC3 );
	while ( 1 )
	{
		frame.copyTo(p_frame);
		n_frame.copyTo(frame);
		capture.read(n_frame);
		Mat pbwf,bwf,nbwf;
		if( n_frame.empty() )
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		cvtColor(p_frame, pbwf, cv::COLOR_BGR2GRAY);
    		cvtColor(frame, bwf, CV_RGB2GRAY);
    		cvtColor(n_frame, nbwf, cv::COLOR_BGR2GRAY);
		//-- 3. Apply the classifier to the frame
		detectAndDisplay( frame );
		for(size_t i=0;i<faces.size();i++)
		{
			rectangle(frame,Point(faces[i].x-faces[i].width,faces[i].y-faces[i].height),Point(faces[i].x+(faces[i].width)*2,faces[i].y+(faces[i].height)*2),Scalar(255,0,255),4,8,0);

		}
		imshow( "initial", frame);
		if((int)faces.size()!=0)
		{
			Rect roi(max(0,faces[0].x-faces[0].width),max(0,faces[0].y-faces[0].height), 3*faces[0].width, 3*faces[0].height);
			Mat image_roi=frame.clone();
                        Mat diff=diffImage(pbwf,bwf,nbwf);
			image_roi=image_roi(roi);
			Size s(128,128);
			resize(image_roi,image_roi,s);
			Mat motion;
                        motion=diff.clone();
			motion=motion(roi);
			resize(motion,motion,s);
			Mat imgHSV;

			cvtColor(image_roi, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

			inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
			threshold(motion, motion, 25, 255, CV_THRESH_BINARY);
			erode(motion, motion, getStructuringElement(MORPH_RECT, Size(2,2)));
			//dilate(motion, motion, getStructuringElement(MORPH_RECT, Size(2,2)) );
			//morphological opening (remove small objects from the foreground)
			erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
			dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

			//morphological closing (fill small holes in the foreground)
			dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
			erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

			bitwise_not ( imgThresholded,imgThresholded  );
			
			if(countNonZero(imgThresholded)<15000&&countNonZero(motion)==0)
                        {
			    IplImage cimage;
			    cimage=imgThresholded;
			    int r=imageClassify.classify(&cimage,1);
			    imline= Mat::zeros( frame.size(), CV_8UC3 );
                            switch(r)
                            {
                               case 0 : putText(imline, "A", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 1 : putText(imline, "B", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 2 : putText(imline, "C", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 3 : putText(imline, "D", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 4 : putText(imline, "E", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 5 : putText(imline, "F", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 6 : putText(imline, "G", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 7 : putText(imline, "H", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 8 : putText(imline, "I", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;
			       case 9 : putText(imline, "J", cvPoint(120,300), FONT_HERSHEY_COMPLEX_SMALL, 20.0, cvScalar(200,200,250), 1, CV_AA); break;

                              }				
				}
                        imshow("Thresholded Image", imgThresholded);
			imshow("motion",motion);
                        imshow("char",imline);
			flag=1;
			//sleep(0.2);
			//cout<<countNonZero(imgThresholded)<<" "<<countNonZero(motion)<<endl;
		}
		else
		{
			if(flag==1)
			{
				destroyWindow("ROI");
				flag=0;
				destroyWindow("Thresholded Image");
				destroyWindow("motion");
				destroyWindow("char");
			}
		}
		int c = waitKey(1);
		if( (char)c == 27 ) { break; } // escape
		if((char)c =='r' || (char) c =='R')
			faces.clear();
		if( (char)c == 'a')
		{
			imwrite("shank.jpeg",imgThresholded);
		}
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces2;
	Mat frame_gray;
	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	//face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
	if((int)faces.size()==0)
		face_cascade.detectMultiScale( frame_gray, faces, 1.25,4,0,Size(0,0), Size(100,100) );
	/*if((int)faces.size()!=0)
	  for ( size_t i = 0; i < faces.size(); i++ )
	  {
	//Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
	//ellipse( frame, center, Size( faces[i].width*1.5, faces[i].height*1.5 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
	rectangle(frame,Point(faces[i].x-faces[i].width,faces[i].y-faces[i].height),Point(faces[i].x+(faces[i].width)*2,faces[i].y+(faces[i].height)*2),Scalar(255,0,255),4,8,0);
	//cout<<"HI";


	//-- In each face, detect eyes
	/*eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

	for ( size_t j = 0; j < eyes.size(); j++ )
	{
	Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
	int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
	circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
	}*/
	//}
	/*else
	  {
	  }
	//-- Show what you got
	imshow( window_name, frame+imline );*/
}
