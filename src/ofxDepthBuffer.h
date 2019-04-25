#pragma once

#include "MSAOpenCL.h"

using namespace msa;

//////////////////////////////////////////////////
// DEPTH BUFFER
//
// Base class for OpenCL-OpenGL interop buffer
class ofxDepthBuffer {
public:
	virtual int getBytesPerType() const = 0;
	virtual int getBytesPerElement() const = 0;
	virtual int getNumElements() const = 0;
	virtual int getNumTypePerElement() const = 0;
	OpenCLBuffer & getCLBuffer();
	ofBufferObject & getGLBuffer();

	void allocate(int numElements);
	bool isAllocated() const;

protected:
	void write(void * data, int numElements);
	void read(void * data, int numElements);
	void copy(ofxDepthBuffer & dest);

	//private:
	ofBufferObject glBuf;
	OpenCLBuffer clBuf;
};

template<typename T, class E = T>
class ofxDepthBufferT : public ofxDepthBuffer {
public:
	int getBytesPerType() const {
		return sizeof(T);
	}
	int getBytesPerElement() const {
		return sizeof(E);
	}
	int getNumElements() const {
		return glBuf.size() / sizeof(E);
	}
	int getNumTypePerElement() const {
		return sizeof(E) / sizeof(T);
	}
	void allocate(int numElements) {
		ofxDepthBuffer::allocate(numElements);
	}
	void write(T * data, int numElements) {
		ofxDepthBuffer::write((void*)data, numElements);
	}
	void read(T * data, int numElements) {
		ofxDepthBuffer::read((void*)data, numElements);
	}
	void copy(ofxDepthBufferT<T,E> & dest) {
		ofxDepthBuffer::copy(dest);
	}
protected:
};