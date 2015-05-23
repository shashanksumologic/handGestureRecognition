Basic Hand gesture recognition application 
==========================================

Basic Hand gesture recognition application  using KNN and openCV

This source code is not for professional uses,  this is for educational uses, and describe the common technics for create an OCR application

Requirements
============

- OpenCV
- CMake

Compilation
===========

    mkdir build
    cd build
    cmake ../imageClassify
    make

Running demo
============

./handGesture

1) Web cam window opens. It needs to detect a close fist to decribe a ROI for hand gestures.
2) once a fist is detected a pink box is made around it!
3) Now the gestures are detected in this ROI.
4) HAnd segmentation is done using the skin color detection and hence may need adjustment with HSV tracker.
5) Motion window shows any pixels which are not same in consecutive frames. The gestures are detected only with 0 white pixels.
6) ROI box is helpful in setting the trackers. It shows the binary image on which gesture recognition is performed .

Keys control
============

    "r" - resets ROI box
    "Esc" - exits the program
    

