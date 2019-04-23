#pragma once

#include "ofxDepthBuffer.h"
#include "MSAOpenCL.h"

using namespace msa;

class ofxDepthData;
class ofxDepthTable;

//////////////////////////////////////////////////
// DEPTH POINTS

template<typename T, class E>
class ofxDepthPointsT : public ofxDepthBufferT<T,E> {
public:
	void write(vector<E> & v) {
		write(v.data(), v.size());
	}
protected:
};

//////////////////////////////////////////////////
// DEPTH POINTS

class ofxDepthPoints : public ofxDepthPointsT<float, ofVec4f> {
public:

	void allocate(int numVertices);

	void load(string filepath);

	void read(ofxDepthData & data);
	void read(vector<ofVec4f> & points);
	void write(ofxDepthData & data);
	void write(vector<ofVec4f> & points, int count);

	void updateMesh(int width, int height, float noiseThreshold = 10.f, bool mode = false);
	void updateTexCoords(ofxDepthTable & table, float u = 1.f, float v = 1.f);
	void updateTexCoords(float x, float y, float scaleX = 1.f, float scaleY = 1.f);

	void draw();
	void drawMesh();

	void smoothNormals(int width, int height);
	void transform(const ofMatrix4x4 & mat);
	void transform(const ofMatrix4x4 & mat, ofxDepthPoints & outputPoints);

	static ofMesh makeFrustum(float fovH, float fovV, float clipNear, float clipFar);

protected:
	static OpenCLKernelPtr getKernel(string name);
	static OpenCLProgramPtr getProgram();
	static OpenCLProgramPtr program;

	ofxDepthBufferT<ofIndexType,ofIndexType> indBuf;
	ofxDepthBufferT<float, ofVec4f> norBuf;
	ofxDepthBufferT<float, ofVec4f> norBufTemp;
	ofxDepthBufferT<float, ofVec4f> colBuf;
	ofxDepthBufferT<float, ofVec2f> texBuf;
	ofVbo vbo;

	OpenCLBufferManagedT<ofVec4f> matrix;
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

	int countZeros();
	int removeZeros(bool resize = true);
	int removeZeros(bool resize, ofxDepthData & outputData);

protected:
	vector<ofVec4f> data;
	int count;
};