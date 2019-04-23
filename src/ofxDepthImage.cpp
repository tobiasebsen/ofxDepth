#include "ofxDepthCore.h"
#include "ofxDepthPoints.h"
#include "ofxDepthImage.h"

#define STRINGIFY(A) #A

string depthImageProgram = STRINGIFY(

	__kernel void flipH(__global unsigned short* input, __global unsigned short* output) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int width = get_global_size(0);
	if (coords.x < width/2) {
		int i = coords.y * width + coords.x;
		int j = coords.y * width + (width - 1 - coords.x);
		unsigned short temp = input[j];
		output[j] = input[i];
		output[i] = temp;
	}
}

__kernel void flipV(__global unsigned short* input, __global unsigned short* output) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int width = get_global_size(0);
	int height = get_global_size(1);
	if (coords.y < height/2) {
		int i = coords.y * width + coords.x;
		int j = (height-coords.y-1) * width + coords.x;
		unsigned short temp = input[j];
		output[j] = input[i];
		output[i] = temp;
	}
}

__kernel void limit(__global unsigned short* input, __global unsigned short* output, int min, int max) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
	if (input[i] < min || input[i] > max)
		output[i] = 0;
	else
		output[i] = input[i];
}

__kernel void denoise(__global unsigned short* input, __global unsigned short* output, float threshold, int neighbours) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int width = get_global_size(0);
	int height = get_global_size(1);
	int i = coords.y * width + coords.x;

	int xmin = coords.x >= 2 ? coords.x-2 : 0;
	int xmax = coords.x < width-2 ? coords.x+2 : width-2;
	int ymin = coords.y >= 2 ? coords.y-2 : 0;
	int ymax = coords.y < height-2 ? coords.y+2 : height-2;

	int sum = 0;

	for (int y=ymin; y<=ymax; y++) {
		for (int x=xmin; x<=xmax; x++) {
			int j = y * width + x;
			if (i != j && abs(input[i] - input[j]) < threshold) {
				sum++;
			}
		}
	}
	if (sum < neighbours)
		output[i] = 0;
	else
		output[i] = input[i];
}

__kernel void erode(__global unsigned short* input, __global unsigned short* output, int radius, float threshold, float fov) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int width = get_global_size(0);
	int i = coords.y * width + coords.x;
	unsigned short c = input[i];

	if (c == 0) {
		output[i] = 0;
		return;
	}

	int diam = radius*2+1;
	float zrt = fabs(c * fov) * threshold;
	int count = 0;
	for (int y=-radius; y<=radius; y++) {
		int iy = (coords.y+y)*width;
		for (int x=-radius; x<=radius; x++) {
			int j = iy+coords.x+x;
			if (input[j] != 0 && abs(input[j] - c) < zrt) {
				count++;
			}
		}
	}
	if (count < (diam*diam)/2)
		output[i] = 0;
	else
		output[i] = c;
}

__kernel void dilate(__global unsigned short* input, __global unsigned short* output, int radius, float threshold, float fov) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int width = get_global_size(0);
	int i = coords.y * width + coords.x;
	unsigned short c = input[i];

	if (c != 0) {
		output[i] = c;
		return;
	}

	int diam = radius*2+1;
	int count = 0;
	unsigned long sum = 0;
	for (int y=-radius; y<=radius; y++) {
		int iy = (coords.y+y)*width;
		for (int x=-radius; x<=radius; x++) {
			int j = iy+coords.x+x;
			if (input[j] != 0) {
				count++;
				sum += input[j];
			}
		}
	}
	unsigned short avg = sum / count;
	float zrt = fabs(avg * fov) * threshold;

	sum = 0;
	for (int y=-radius; y<=radius; y++) {
		int iy = (coords.y+y)*width;
		for (int x=-radius; x<=radius; x++) {
			int j = iy+coords.x+x;
			if (input[j] != 0) {
				unsigned long d = input[j] - avg;
				sum += d * d;
			}
		}
	}

	if (count >= (diam*diam)/2-1 && sqrt((double)sum / count) < zrt)
		output[i] = avg;
	else
		output[i] = 0;
}

__constant float gaus0 = 0.077847;
__constant float gaus1 = 0.123317;
__constant float gaus2 = 0.195346;

__kernel void blur(__global unsigned short* input, __global unsigned short* output) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);

	if (x < 1 || x >= width-1)
		return;
	if (y < 1 || y >= height-1)
		return;

	int y0 = (y-1) * width;
	int y1 = (y+0) * width;
	int y2 = (y+1) * width;

	double avg = 0;
	avg += input[y0+x-1] * gaus0;
	avg += input[y0+x+0] * gaus1;
	avg += input[y0+x+1] * gaus0;
	avg += input[y1+x-1] * gaus1;
	avg += input[y1+x+0] * gaus2;
	avg += input[y1+x+1] * gaus1;
	avg += input[y2+x-1] * gaus0;
	avg += input[y2+x+0] * gaus1;
	avg += input[y2+x+1] * gaus0;

	output[y1+x] = (unsigned short)avg;
}

__kernel void convolution(__global unsigned short* input, __global unsigned short* output, __global float * ker, int radius, int threshold, float fov) {
	int cx = get_global_id(0);
	int cy = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int i = cy*width+cx;
	unsigned short c = input[i];

	if (c == 0) {
		output[i] = 0;
		return;
	}

	int diam = radius*2+1;
	float zrt = fabs(c * fov) * threshold;
	double avg = 0;
	for (int y=-radius; y<=radius; y++) {
		int iy = (cy+y)*width;
		int ky = (radius+y)*diam;
		for (int x=-radius; x<=radius; x++) {
			int kx = radius+x;
			int j = iy+cx+x;
			if (input[j] != 0 && abs(input[j] - input[i]) < zrt)
				avg += input[j] * ker[ky+kx];
			else
				avg += c * ker[ky+kx];
		}
	}
	output[i] = (unsigned short)avg;
}

__kernel void map(__global unsigned short* depthIn, unsigned short imin, unsigned short imax, unsigned short omin, unsigned short omax, __global unsigned short* depthOut) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
	if (depthIn[i] == 0)
		depthOut[i] = 0;
	else
		depthOut[i] = (( (int)depthIn[i] - imin) * ( (int)omax - omin)) / ( (int)imax - imin) + omin;
}

__kernel void accumulate(__global unsigned short* input, __global unsigned short* output, float amount, int threshold) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
	if (input[i] > 0) {
		if (output[i] == 0 || abs(input[i] - output[i]) > threshold)
			output[i] = input[i];
		else
			output[i] = output[i] * (1-amount) + input[i] * amount;
	}
}

__kernel void stabilize(__global unsigned short* input, __global unsigned short* mean, __global float* variance, __global unsigned short* output, float amount, float threshold) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
	mean[i] = mean[i] * (1-amount) + input[i] * amount;
	float d = input[i] - mean[i];
	d /= 8000.0f;
	float var = d * d;
	variance[i] = variance[i] * (1-amount) + var * amount;
	if (var / sqrt(variance[i]) < threshold)
		output[i] = mean[i];
	else
		output[i] = input[i];
}

__kernel void subtract(__global unsigned short* input, __global unsigned short* background, __global unsigned short* output, int threshold) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
	if (background[i] == 0 || background[i] - input[i] > threshold)
		output[i] = input[i];
	else
		output[i] = 0;
}

__kernel void pointsFromFov(__global unsigned short* depth, float2 fov, __global float4* points) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dims = (int2)(get_global_size(0), get_global_size(1));
	float2 angle = fov * (((float2)(coords.x, coords.y) / (float2)(dims.x, dims.y)) - (float2)(0.5f));
	int i = coords.y * dims.x + coords.x;
	points[i].x = tan(radians(angle.x)) * depth[i];
	points[i].y = tan(radians(angle.y)) * depth[i];
	points[i].z = -depth[i];
	points[i].w = 1.f;
}

__kernel void pointsFromTable(__global unsigned short* depth, __global float2* table, __global float4* points) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
	float d = (float)depth[i];
	points[i].x = table[i].x * d;
	points[i].y = table[i].y * d;
	points[i].z = -d;
	points[i].w = 1.f;
}
);

//////////////////////////////////////////////////

OpenCLProgramPtr ofxDepthImage::program;

void ofxDepthImage::load(string filepath) {
	ofxDepth.setup();
	ofShortPixels pixels;
	ofLoadImage(pixels, filepath);
	ofShortPixels spixels = pixels.getChannel(0);
	write(spixels);
}

void ofxDepthImage::updateTexture() {
	ofxDepthImageT<unsigned short>::update(tex);    
}

void ofxDepthImage::draw(float x, float y) {
	if (tex.isAllocated())
		tex.draw(x, y);
}

void ofxDepthImage::draw(float x, float y, float w, float h) {
	if (tex.isAllocated())
		tex.draw(x, y, w, h);
}

void ofxDepthImage::flipHorizontal() {
	OpenCLKernelPtr kernel = ofxDepth.getKernel("flipH");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, getCLBuffer());
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::flipVertical() {
	OpenCLKernelPtr kernel = ofxDepth.getKernel("flipV");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, getCLBuffer());
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::limit(int min, int max) {
	OpenCLKernelPtr kernel = ofxDepth.getKernel("limit");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, getCLBuffer());
	kernel->setArg(2, min);
	kernel->setArg(3, max);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::denoise(float threshold, int neighbours, ofxDepthImage &outputImage) {

	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = ofxDepth.getKernel("denoise");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, outputImage.getCLBuffer());
	kernel->setArg(2, threshold);
	kernel->setArg(3, neighbours);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::denoise(float threshold, int neighbours) {
	denoise(threshold, neighbours, *this);
}

void ofxDepthImage::erode(int radius, float threshold, ofxDepthImage &outputImage) {

	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = getKernel("erode");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, outputImage.getCLBuffer());
	kernel->setArg(2, radius);
	kernel->setArg(3, threshold);
	kernel->setArg(4, tanf(60*DEG_TO_RAD)/height);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::dilate(int radius, float threshold, ofxDepthImage &outputImage) {

	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = getKernel("dilate");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, outputImage.getCLBuffer());
	kernel->setArg(2, radius);
	kernel->setArg(3, threshold);
	kernel->setArg(4, tanf(60*DEG_TO_RAD)/height);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::blur(ofxDepthImage & outputImage) {

	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = getKernel("blur");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, outputImage.getCLBuffer());
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::convolution(OpenCLBufferManagedT<float> & conv, int radius, ofxDepthImage & outputImage) {

	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = getKernel("convolution");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, outputImage.getCLBuffer());
	kernel->setArg(2, conv);
	kernel->setArg(3, radius);
	kernel->setArg(4, 5);
	kernel->setArg(5, tanf(60 * DEG_TO_RAD)/height);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin, uint16_t outputMax, ofxDepthImage & outputImage) {

	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = getKernel("map");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, &inputMin, sizeof(uint16_t));
	kernel->setArg(2, &inputMax, sizeof(uint16_t));
	kernel->setArg(3, &outputMin, sizeof(uint16_t));
	kernel->setArg(4, &outputMax, sizeof(uint16_t));
	kernel->setArg(5, outputImage.getCLBuffer());
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::accumulate(ofxDepthImage & outputImage, float amount, int threshold) {

	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = getKernel("accumulate");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, outputImage.getCLBuffer());
	kernel->setArg(2, amount);
	kernel->setArg(3, threshold);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::stabilize(ofxDepthImage & meanImage, ofxDepthImageT<float>& varImage, ofxDepthImage & outputImage, float amount, float threshold) {

	if (!meanImage.isAllocated())
		meanImage.allocate(getWidth(), getHeight());
	if (!varImage.isAllocated())
		varImage.allocate(getWidth(), getHeight());
	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = getKernel("stabilize");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, meanImage.getCLBuffer());
	kernel->setArg(2, varImage.getCLBuffer());
	kernel->setArg(3, outputImage.getCLBuffer());
	kernel->setArg(4, amount);
	kernel->setArg(5, threshold);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::subtract(ofxDepthImage & background, int threshold) {

	if (!background.isAllocated())
		return;

	OpenCLKernelPtr kernel = getKernel("subtract");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, background.getCLBuffer());
	kernel->setArg(2, getCLBuffer());
	kernel->setArg(3, threshold);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin, uint16_t outputMax) {
	map(inputMin, inputMax,  outputMin, outputMax, *this);
}

void ofxDepthImage::toPoints(float fovH, float fovV, ofxDepthPoints & points) {

	if (!points.isAllocated())
		points.allocate(getNumElements());

	OpenCLKernelPtr kernel = getKernel("pointsFromFov");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, ofVec2f(fovH, fovV));
	kernel->setArg(2, points.getCLBuffer());
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::toPoints(ofxDepthTable & table, ofxDepthPoints & points) {

	if (!points.isAllocated())
		points.allocate(getNumElements());

	OpenCLKernelPtr kernel = getKernel("pointsFromTable");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, table.getCLBuffer());
	kernel->setArg(2, points.getCLBuffer());
	kernel->run2D(getWidth(), getHeight());
}

OpenCLKernelPtr ofxDepthImage::getKernel(string name) {
	getProgram();
	return ofxDepth.getKernel(name);
}

OpenCLProgramPtr ofxDepthImage::getProgram() {
	if (program)
		return program;
	else {
		program = ofxDepth.loadProgram(depthImageProgram);
		ofxDepth.loadKernel("flipH", program);
		ofxDepth.loadKernel("flipV", program);
		ofxDepth.loadKernel("limit", program);
		ofxDepth.loadKernel("denoise", program);
		ofxDepth.loadKernel("erode", program);
		ofxDepth.loadKernel("dilate", program);
		ofxDepth.loadKernel("blur", program);
		ofxDepth.loadKernel("convolution", program);
		ofxDepth.loadKernel("map", program);
		ofxDepth.loadKernel("accumulate", program);
		ofxDepth.loadKernel("stabilize", program);
		ofxDepth.loadKernel("subtract", program);
		ofxDepth.loadKernel("pointsFromFov", program);
		ofxDepth.loadKernel("pointsFromTable", program);
		return program;
	}
}
