#include "ofxDepthCore.h"

//////////////////////////////////////////////////

void ofxDepthCore::setup(int deviceNumber) {
	if (isSetup())
		return;
	opencl.setupFromOpenGL(deviceNumber);
	//loadKernels();
}

bool ofxDepthCore::setup(string vendorName, string deviceName) {
	if (isSetup())
		return true;
	int n = opencl.getDeviceInfos();
	for (int i=0; i<n; i++) {
		if ((string((char*)opencl.deviceInfo[i].vendorName).find(vendorName) != string::npos && deviceName == "") ||
			string((char*)opencl.deviceInfo[i].deviceName) == deviceName) {
			ofLog() << "Setting up OpenCL...";
			ofLog() << "Vendor: " << opencl.deviceInfo[i].vendorName;
			ofLog() << "Device: " << opencl.deviceInfo[i].deviceName;
			setup(i);
			return true;
		}
	}
	return false;
}

bool ofxDepthCore::isSetup() {
	return opencl.getDevice() != nullptr;
}

void ofxDepthCore::loadKernel(string name, OpenCLProgramPtr program) {
	opencl.loadKernel(name, program);
}

/*void ofxDepthCore::loadKernels() {
	opencl.loadProgramFromSource(depthImageProgram);
	opencl.loadKernel("flipH");
	opencl.loadKernel("flipV");
	opencl.loadKernel("limit");
	opencl.loadKernel("map");
	opencl.loadKernel("accumulate");
	opencl.loadKernel("stabilize");
	opencl.loadKernel("subtract");
	opencl.loadKernel("pointsFromFov");
	opencl.loadKernel("pointsFromTable");
	opencl.loadKernel("denoise");
	opencl.loadKernel("erode");
	opencl.loadKernel("dilate");
	opencl.loadKernel("blur");
	opencl.loadKernel("convolution");

	opencl.loadKernel("pointsToIndices");
	opencl.loadKernel("smoothNormals");
	opencl.loadKernel("calcNormals");
	opencl.loadKernel("transform");
	opencl.loadKernel("mapTexCoords");
	opencl.loadKernel("orthTexCoords");
}*/

OpenCL & ofxDepthCore::getCL() {
	setup();
	return opencl;
}

OpenCLProgramPtr ofxDepthCore::loadProgram(string source) {
	return opencl.loadProgramFromSource(source);
}

OpenCLKernelPtr ofxDepthCore::getKernel(string name) {
	return getCL().kernel(name);
}
