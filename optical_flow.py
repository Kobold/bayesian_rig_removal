"""
The rest of the example code that I ganked.

// 3. Draw arrows on second image to show movement
for (int x=0; x<frame2->width; x=x+10) {
  for (int y=0; y<frame2->height; y=y+10) {
    int vel_x_here = (int)cvGetReal2D( vel_x, y, x);
    int vel_y_here = (int)cvGetReal2D( vel_y, y, x);
    cvLine( frame2, cvPoint(x, y), cvPoint(x+vel_x_here,
y+vel_y_here), cvScalarAll(255));
  }
}
		
// 4. Display image with arrows and wait for key to be pressed
cvShowImage("result", frame2 );
cvWaitKey(0);
"""
from cv import CalcOpticalFlowLK, CreateImage, GetReal2D, IPL_DEPTH_32F, LoadImage, \
               CV_LOAD_IMAGE_GRAYSCALE
import csv

# load images
frame1 = LoadImage('/Users/kobold/Desktop/forest_start/forest 001.png', CV_LOAD_IMAGE_GRAYSCALE)
frame2 = LoadImage('/Users/kobold/Desktop/forest_start/forest 002.png', CV_LOAD_IMAGE_GRAYSCALE)

# calculate optical flow
vel_x = CreateImage((frame1.width, frame1.height), IPL_DEPTH_32F, 1)
vel_y = CreateImage((frame1.width, frame1.height), IPL_DEPTH_32F, 1)
CalcOpticalFlowLK(frame2, frame1, (5, 5), vel_x, vel_y)

# dump the pixel velocities
x_writer = csv.writer(open('vel_x.csv', 'wb'))
y_writer = csv.writer(open('vel_y.csv', 'wb'))
for y in xrange(frame1.height):
    x_writer.writerow([GetReal2D(vel_x, y, x) for x in xrange(frame1.width)])
    y_writer.writerow([GetReal2D(vel_y, y, x) for x in xrange(frame1.width)])
