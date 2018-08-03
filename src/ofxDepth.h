#pragma once

#include "MSAOpenCL.h"

using namespace msa;

class ofxDepth {
public:
	void setup(int width = 0, int height = 0, int deviceNumber = -1);

	void write(ofShortPixels & pixels);
	void read(ofShortPixels & pixels);
    
    void drawPoints();

	void updateTexture(ofTexture & tex);

	void map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin = 0, uint16_t outputMax = USHRT_MAX);
    void toPoints(float fovH, float fovV, OpenCLBuffer & points);
    void toPoints(float fovH, float fovV);

private:
	OpenCL opencl;
	ofBufferObject texObj;
    ofBufferObject pntObj;
	OpenCLBuffer depth;
    OpenCLBuffer points;
	int width;
	int height;
};