#pragma once

#include "MSAOpenCL.h"

using namespace msa;

//////////////////////////////////////////////////
// DEPTH CORE

class ofxDepthCore {
public:
	static ofxDepthCore & get() {
		static ofxDepthCore core;
		return core;
	}

	void setup(int deviceNumber = -1);
	bool setup(string vendorName, string deviceName = "");
	bool isSetup();
	void loadKernel(string name, OpenCLProgramPtr program);

	OpenCL & getCL();
	OpenCLProgramPtr loadProgram(string source);
	OpenCLKernelPtr getKernel(string name);

private:
	ofxDepthCore() {}
	OpenCL opencl;
};

static ofxDepthCore & ofxDepth = ofxDepthCore::get();
