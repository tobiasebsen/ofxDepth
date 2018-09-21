#pragma once

#include "MSAOpenCL.h"

using namespace msa;

class ofxDepthPoints;

//////////////////////////////////////////////////
// DEPTH TABLE
//
// Depth (z) to world position table

class ofxDepthTable {
public:
    void allocate(int width, int height);
	bool isAllocated();
    
    void write(ofFloatPixels & entries);

    OpenCLBuffer & getCLBuffer() {
        return clBuf;
    }

protected:
    OpenCLBuffer clBuf;
};


//////////////////////////////////////////////////
// DEPTH IMAGE
//
// The raw image from a depth camera

class ofxDepthImage {
public:

    void allocate(int width, int height);
    bool isAllocated() const;
    
    int getWidth() const {
        return width;
    }
    int getHeight() const {
        return height;
    }
    int getNumPixels() const {
        return width * height;
    }
    
    void load(string filepath);

    void write(ofShortPixels & pixels);
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

    OpenCLBuffer & getCLBuffer() {
        return clBuf;
    }

protected:
    int width;
    int height;
    ofBufferObject glBuf;
    OpenCLBuffer clBuf;
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

class ofxDepthPoints {
public:

    void allocate(int size);
    bool isAllocated();
    
    int getSize() const {
        return size;
    }
    int getCount() const {
        return count;
    }
    void setCount(int count) {
        this->count = count;
    }
    
    void load(string filepath);

    void read(ofxDepthData & data);
    void read(vector<ofVec4f> & points);
    void write(ofxDepthData & data);
    void write(vector<ofVec4f> & points);
    void write(vector<ofVec4f> & points, int count);
    
    void draw();
    
    void transform(const ofMatrix4x4 & mat);
    void transform(const ofMatrix4x4 & mat, ofxDepthPoints & outputPoints);
    
	static ofMesh makeFrustum(float fovH, float fovV, float clipNear, float clipFar);
	
	OpenCLBuffer & getCLBuffer() {
        return clBuf;
    }

protected:
    int size;
    int count;
    ofBufferObject glBuf;
    OpenCLBuffer clBuf;
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
