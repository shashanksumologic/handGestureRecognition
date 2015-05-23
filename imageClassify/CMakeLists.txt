PROJECT( handGesture )
FIND_PACKAGE( OpenCV REQUIRED )

ADD_EXECUTABLE( handGesture main.cpp ImageClassify.cpp preprocessing.cpp )
TARGET_LINK_LIBRARIES( handGesture ${OpenCV_LIBS} )
