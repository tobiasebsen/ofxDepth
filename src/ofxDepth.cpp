#include "ofxDepth.h"

#define STRINGIFY(A) #A

string depthProgram = STRINGIFY(
__kernel void map(__global unsigned short* depth, unsigned short imin, unsigned short imax, unsigned short omin, unsigned short omax) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
	depth[i] = (depth[i] - imin) * ((omax - omin) / (imax - imin)) + omin;
}

__kernel void toPoints(__global unsigned short* depth, float2 fov, __global float4* points) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    int2 dims = (int2)(get_global_size(0), get_global_size(1));
    float2 angle = fov * (((float2)coords / dims) - float2(0.5f));
    int i = coords.y * dims.x + coords.x;
    points[i].x = tan(radians(angle.x)) * depth[i];
    points[i].y = tan(radians(angle.y)) * depth[i];
    points[i].z = depth[i];
    points[i].w = 0;
}
);

void ofxDepth::setup(int width, int height, int deviceNumber) {
	this->width = width;
	this->height = height;
	opencl.setupFromOpenGL(deviceNumber);
	opencl.loadProgramFromSource(depthProgram);
	opencl.loadKernel("map");
    opencl.loadKernel("toPoints");
	if (width != 0 && height != 0) {
		texObj.allocate(width * height * sizeof(unsigned short), GL_STREAM_DRAW);
		depth.initFromGLObject(texObj.getId());
        pntObj.allocate(width * height * sizeof(ofVec4f), GL_STREAM_DRAW);
        points.initFromGLObject(pntObj.getId());
	}
}

void ofxDepth::write(ofShortPixels & p) {
	if (width == 0 || height == 0) {
		texObj.allocate(p.getWidth() * p.getHeight() * sizeof(unsigned short), GL_STREAM_DRAW);
		depth.initFromGLObject(texObj.getId());
		//depth.initBuffer(p.getWidth() * p.getHeight() * sizeof(unsigned short));
	}
	width = p.getWidth();
	height = p.getHeight();
	depth.write(p.getData(), 0, width * height * p.getBytesPerPixel());
}

void ofxDepth::read(ofShortPixels & p) {
	depth.read(p.getData(), 0, p.getWidth() * p.getHeight() * p.getBytesPerPixel());
}

void ofxDepth::updateTexture(ofTexture & tex) {
	texObj.bind(GL_PIXEL_UNPACK_BUFFER);
	tex.loadData((unsigned short*)NULL, width, height, GL_LUMINANCE);
	texObj.unbind(GL_PIXEL_UNPACK_BUFFER);
}

void ofxDepth::map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin, uint16_t outputMax) {
	OpenCLKernelPtr kernel = opencl.kernel("map");
	kernel->setArg(0, depth);
	kernel->setArg(1, &inputMin, sizeof(uint16_t));
	kernel->setArg(2, &inputMax, sizeof(uint16_t));
	kernel->setArg(3, &outputMin, sizeof(uint16_t));
	kernel->setArg(4, &outputMax, sizeof(uint16_t));
	kernel->run2D(width, height);
}

void ofxDepth::toPoints(float fovH, float fovV, OpenCLBuffer & points) {
    OpenCLKernelPtr kernel = opencl.kernel("toPoints");
    kernel->setArg(0, depth);
    kernel->setArg(1, ofVec2f(fovH, fovV));
    kernel->setArg(2, points);
    kernel->run2D(width, height);
}

void ofxDepth::toPoints(float fovH, float fovV) {
    if (!pntObj.isAllocated()) {
        pntObj.allocate(width * height * sizeof(ofVec4f), GL_STREAM_DRAW);
        points.initFromGLObject(pntObj.getId());
    }
    toPoints(fovH, fovV, points);
}
