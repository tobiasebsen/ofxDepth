#pragma once

#include "MSAOpenCL.h"

using namespace msa;

class ofxDepthPoints;

//////////////////////////////////////////////////
// DEPTH BUFFER
//
// Base class for OpenCL-OpenGL interop buffer
class ofxDepthBuffer {
public:
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

	ofBufferObject glBuf;
	OpenCLBuffer clBuf;
};

template<typename T, class E>
class ofxDepthBufferT : public ofxDepthBuffer {
public:
	int getBytesPerElement() const {
		return sizeof(E);
	}
	int getNumElements() const {
		return glBuf.size() / sizeof(T);
	}
	int getNumTypePerElement() const {
		return sizeof(E) / sizeof(T);
	}
	void allocate(int numElements) {
		ofxDepthBuffer::allocate(numElements);
	}
	void allocate(int width, int height) {
		ofxDepthBuffer::allocate(width * height);
		dimSize[0] = width;
		dimSize[1] = height;
	}
	void write(T * data, int numElements) {
		ofxDepthBuffer::write((void*)data, numElements);
		dimSize[0] = numElements;
	}
	void write(vector<E> & v) {
		write(v.data(), v.size());
	}
	void write(ofPixels_<T> & p) {
		write(p.getData(), p.getTotalBytes() / p.getBytesPerPixel());
		dimSize[0] = p.getWidth();
		dimSize[1] = p.getHeight();
	}
protected:
	int dimSize[3];
};

//////////////////////////////////////////////////
// DEPTH TABLE
//
// Depth (z) to world position table

class ofxDepthTable : public ofxDepthBufferT<float, ofVec2f> {
public:
    void allocate(int width, int height);
};


//////////////////////////////////////////////////
// DEPTH IMAGE
//
// The raw image from a depth camera

class ofxDepthImage : public ofxDepthBufferT<unsigned short, unsigned short> {
public:
    
    int getWidth() const {
        return dimSize[0];
    }
    int getHeight() const {
        return dimSize[1];
    }
    
    void load(string filepath);

    //void write(ofShortPixels & pixels);
    void read(ofShortPixels & pixels);
    
    void update();
    void update(ofTexture & tex);
    
    void draw(float x, float y);
    void draw(float x, float y, float w, float h);
    
	void flipHorizontal();
	void flipVertical();
	void limit(int min, int max);
    void denoise(float threshold, int neighbours = 1);
    void denoise(float threshold, int neighbours, ofxDepthImage & outputImage);
    void map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin = 0, uint16_t outputMax = USHRT_MAX);
    void map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin, uint16_t outputMax, ofxDepthImage & outputImage);
	void accumulate(ofxDepthImage & outputImage, float amount);
	void subtract(ofxDepthImage & background, int threshold);

    void toPoints(float fovH, float fovV, ofxDepthPoints & points);
    void toPoints(ofxDepthTable & depthTable, ofxDepthPoints & points);

protected:
    ofTexture tex;
};

//////////////////////////////////////////////////
// DEPTH DATA

class ofxDepthData {
public:
    void allocate(int size);
    
    int getSize() const {
        return data.size();
    }
    int getCount() const {
        return count;
    }
    void setCount(int count) {
        this->count = count;
    }

    vector<ofVec4f> & getData() {
        return data;
    }
    
    void getMinMax(ofVec4f & min, ofVec4f & max);

    int countZeros();
    int removeZeros(bool resize = true);
    int removeZeros(bool resize, ofxDepthData & outputData);

protected:
    vector<ofVec4f> data;
    int count;
};


//////////////////////////////////////////////////
// DEPTH POINTS

class ofxDepthPoints : public ofxDepthBufferT<float, ofVec4f> {
public:

    void allocate(int numVertices);
    
    void load(string filepath);

    void read(ofxDepthData & data);
    void read(vector<ofVec4f> & points);
    void write(ofxDepthData & data);
    void write(vector<ofVec4f> & points);
    void write(vector<ofVec4f> & points, int count);

	void updateMesh(int width, int height, float noiseThreshold = 10.f);
    
    void draw();
	void drawMesh();
    
    void transform(const ofMatrix4x4 & mat);
    void transform(const ofMatrix4x4 & mat, ofxDepthPoints & outputPoints);
    
	static ofMesh makeFrustum(float fovH, float fovV, float clipNear, float clipFar);

protected:
	ofxDepthBufferT<ofIndexType,ofIndexType> indBuf;
	ofxDepthBufferT<float, ofVec4f> norBuf;
    ofVbo vbo;
    
    OpenCLBufferManagedT<ofVec4f> matrix;
};

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
	void loadKernels();
    
    OpenCL & getCL();
    OpenCLKernelPtr getKernel(string name);

private:
    ofxDepthCore() {}
	OpenCL opencl;
};

static ofxDepthCore & ofxDepth = ofxDepthCore::get();
