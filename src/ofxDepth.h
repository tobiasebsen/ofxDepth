#pragma once

#include "MSAOpenCL.h"

using namespace msa;

class ofxDepth {
public:
	void setup(int width = 0, int height = 0, int deviceNumber = -1);

	void write(ofShortPixels & pixels);
	void read(ofShortPixels & pixels);

	void updateTexture(ofTexture & tex);

	void map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin = 0, uint16_t outputMax = USHRT_MAX);

private:
	OpenCL opencl;
	ofBufferObject texObj;
	OpenCLBuffer depth;
	int width;
	int height;
};