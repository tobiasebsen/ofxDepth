#include "ofxDepthBuffer.h"


//////////////////////////////////////////////////

OpenCLBuffer & ofxDepthBuffer::getCLBuffer() {
	return clBuf;
}

ofBufferObject & ofxDepthBuffer::getGLBuffer() {
	return glBuf;
}

void ofxDepthBuffer::allocate(int numElements) {
	glBuf.allocate(numElements * getBytesPerElement(), GL_STREAM_DRAW);
	clBuf.initFromGLObject(glBuf.getId());
}

bool ofxDepthBuffer::isAllocated() const {
	return glBuf.isAllocated();
}

void ofxDepthBuffer::write(void * data, int numElements) {
	if (!isAllocated())
		allocate(numElements);
	clBuf.write(data, 0, numElements * getBytesPerElement());
}

void ofxDepthBuffer::read(void * data, int numElements) {
	clBuf.read(data, 0, numElements * getBytesPerElement());
}

void ofxDepthBuffer::copy(ofxDepthBuffer & dest) {
	clBuf.lockGLObject();
	dest.clBuf.copyFrom(clBuf, 0, 0, glBuf.size());
	clBuf.unlockGLObject();
}

