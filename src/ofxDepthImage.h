#pragma once

#include "MSAOpenCL.h"
#include "ofxDepthBuffer.h"

using namespace msa;

class ofxDepthPoints;

template<typename T, class E = T>
class ofxDepthImageT : public ofxDepthBufferT<T,E> {
public:
	int getWidth() {
		return width;
	}
	int getHeight() {
		return height;
	}
	void allocate(int width, int height) {
		ofxDepthBuffer::allocate(width * height);
		this->width = width;
		this->height = height;
	}
	void write(ofPixels_<T> & p) {
		ofxDepthBufferT<T,E>::write(p.getData(), p.getTotalBytes() / p.getBytesPerPixel());
		width = p.getWidth();
		height = p.getHeight();
	}
	void update(ofTexture & tex) {
		glBuf.bind(GL_PIXEL_UNPACK_BUFFER);
		tex.loadData((T*)NULL, getWidth(), getHeight(), GL_LUMINANCE);
		glBuf.unbind(GL_PIXEL_UNPACK_BUFFER);
	}
protected:
	int width;
	int height;
};

//////////////////////////////////////////////////
// DEPTH TABLE
//
// Depth (z) to world position table

class ofxDepthTable : public ofxDepthImageT<float, ofVec2f> {};


//////////////////////////////////////////////////
// DEPTH IMAGE
//
// The raw image from a depth camera

class ofxDepthImage : public ofxDepthImageT<unsigned short> {
public:

	ofTexture & getTexture() {
		return tex;
	}

	void load(string filepath);

	void updateTexture();

	void draw(float x, float y);
	void draw(float x, float y, float w, float h);

	void flipHorizontal();
	void flipVertical();
	void limit(int min, int max);
	void denoise(float threshold, int neighbours = 1);
	void denoise(float threshold, int neighbours, ofxDepthImage & outputImage);
	void erode(int radius, float threshold, ofxDepthImage & outputImage);
	void dilate(int radius, float threshold, ofxDepthImage & outputImage);
	void blur(ofxDepthImage & outputImage);
	void convolution(OpenCLBufferManagedT<float> & convKernel, int radius, ofxDepthImage & outputImage);
	void map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin = 0, uint16_t outputMax = USHRT_MAX);
	void map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin, uint16_t outputMax, ofxDepthImage & outputImage);
	void accumulate(ofxDepthImage & outputImage, float amount, int threshold);
	void stabilize(ofxDepthImage & meanImage, ofxDepthImageT<float> & varImage, ofxDepthImage & outputImage, float amount, float threshold);
	void subtract(ofxDepthImage & background, int threshold);

	void toPoints(float fovH, float fovV, ofxDepthPoints & points);
	void toPoints(ofxDepthTable & depthTable, ofxDepthPoints & points);

protected:
	static OpenCLKernelPtr getKernel(string name);
	static OpenCLProgramPtr getProgram();
	static OpenCLProgramPtr program;

	ofTexture tex;
};