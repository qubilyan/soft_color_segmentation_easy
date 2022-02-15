
// flowIO.h

#ifndef _OpticFlowIO_H
#define _OpticFlowIO_H

#include <math.h>
#include <memory.h>
#include <string.h>

#include <stdint.h>
#include "opencv2/opencv.hpp" // for KITTI

// read and write our simple .flo flow file format

// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

typedef struct 
{
	double aee; // Average Endpoint Error
	double aae; // Average Angular Error
}FlowErr;

class OpticFlowIO
{
public:
	// return whether flow vector is unknown
	template <class T>
	static bool unknown_flow(T u, T v);
	template <class T>
	static bool unknown_flow(T *f);

	// read a flow file into 2-band image
	template <class T>
	static int ReadFlowFile(T* U, T* V, int* w, int* h, const char* filename);

	// write a 2-band image into flow file 
	template <class T>
	static int WriteFlowFile(T* U, T* V, int w, int h, const char* filename);

	// read a KITTI flow file into 2-band image
	template <class T>
	static int ReadKittiFlowFile(T* U, T* V, int* w, int* h, const char* filename);

	// write a 2-band image into KITTI flow file 
	template <class T>
	static int WriteKittiFlowFile(T* U, T* V, int w, int h, const char* filename);

	// render the motion to a 4-band BGRA color image
	template <class T>
	static double MotionToColor(unsigned char* fillPix, T* U, T* V, int w, int h, float range = -1);

	template <class T>
	static float ShowFlow(const char* winname, T* U, T* V, int w, int h, float range = -1, int waittime = 1);
	template <class T>
	static void SaveFlowAsImage(const char* imgName, T* U, T* V, int w, int h, float range = -1);
	static void SaveFlowMatrixAsImage(const char* imgName, cv::Mat2f flowMat);

	static void Match2Flow(FImage& inMat, FImage& ou, FImage& ov, int w, int h);
	static void FlowMatrix2Flow(cv::Mat2f inMat, FImage& ou, FImage& ov);

	template <class T>
	static float ErrorImage(unsigned char* fillPix, T* u1, T* v1, T* u2, T* v2, int w, int h);
	template <class T>
	static float ErrorImage(unsigned char* fillPix, T* u1, T* v1, char* gtName, int w, int h);
	template <class T>
	static float ShowErrorImage(const char* winname, T* U, T* V, char* gtName, int w, int h, int waittime = 1);
	template <class T>
	static float SaveErrorImage(const char* imgName, T* U, T* V, char* gtName, int w, int h);

	template <class T1, class T2>
	static FlowErr CalcFlowError(T1* u1, T1* v1, T2* u2, T2*v2, int w, int h);

private:
	// first four bytes, should be the same in little endian
	#define TAG_FLOAT 202021.25  // check for this when READING the file
	#define TAG_STRING "PIEH"    // use this when WRITING the file