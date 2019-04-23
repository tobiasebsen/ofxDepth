#include "ofxDepthCore.h"
#include "ofxDepthPoints.h"
#include "ofxDepthImage.h"

#define STRINGIFY(A) #A

string depthPointsProgram = STRINGIFY(
	__kernel void transform(__global float4 *input, __global float4 *output, __global float4 *mat) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int width = get_global_size(0);
	int i = coords.y * width + coords.x;
	float4 v = input[i];

	output[i].x = mat[0].x * v.x + mat[1].x * v.y + mat[2].x * v.z + mat[3].x;
	output[i].y = mat[0].y * v.x + mat[1].y * v.y + mat[2].y * v.z + mat[3].y;
	output[i].z = mat[0].z * v.x + mat[1].z * v.y + mat[2].z * v.z + mat[3].z;
	output[i].z = 1.f;
}

__kernel void smoothNormals(__global float4 *input, __global float4 *output) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);

	int yx0 = (y-1) * width + x;
	int yx1 = (y+0) * width + x;
	int yx2 = (y+1) * width + x;

	float4 n1 = input[yx0+0];
	float4 n2 = input[yx1-1];
	float4 n3 = input[yx1+0];
	float4 n4 = input[yx1+1];
	float4 n5 = input[yx2+1];

	if (length(n1) == 0.0f || length(n2) == 0.0f || length(n4) == 0.0f || length(n5) == 0.0f)
		output[yx1] = n3;
	else {
		float4 avg;
		avg += n1 * 0.15f;
		avg += n2 * 0.15f;
		avg += n3 * 0.4f;
		avg += n4 * 0.15f;
		avg += n5 * 0.15f;
		output[yx1] = avg;
	}
}

__kernel void pointsToIndices(__global float4* vertices, __global unsigned int* indices, __global float4* normals, float maxFaceDist, float fov) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	int i = (coords.y * dim.x + coords.x) * 6;

	int c1 = coords.y * dim.x + coords.x;
	int c2 = coords.y * dim.x + coords.x + 1;
	int c3 = (coords.y + 1) * dim.x + coords.x + 1;
	int c4 = (coords.y + 1) * dim.x + coords.x;

	float4 v1 = vertices[c1];
	float4 v2 = vertices[c2];
	float4 v3 = vertices[c3];
	float4 v4 = vertices[c4];

	float zrt1 = fabs(v1.z * fov) * maxFaceDist;
	float zrt3 = fabs(v3.z * fov) * maxFaceDist;

	if (length(v1 - v2) < zrt1 && length(v1 - v4) < zrt1) {
		indices[i+0] = c1;
		indices[i+1] = c4;
		indices[i+2] = c2;
		normals[c1] = normalize(cross(v2-v1, v4-v1));
	}
	else {
		indices[i+0] = 0;
		indices[i+1] = 0;
		indices[i+2] = 0;
	}
	if (length(v3 - v2) < zrt3 && length(v3 - v4) < zrt3) {
		indices[i+3] = c2;
		indices[i+4] = c4;
		indices[i+5] = c3;
	}	
	else {
		indices[i+3] = 0;
		indices[i+4] = 0;
		indices[i+5] = 0;
	}
}

__kernel void calcNormals(__global float4* vertices, __global float4* normals, float maxFaceDist, float fov) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	int c = coords.y * dim.x + coords.x;

	int2 corners[8];
	corners[0] = (int2)(-1,0);
	corners[1] = (int2)(-1,-1);
	corners[2] = (int2)(0,-1);
	corners[3] = (int2)(1,-1);
	corners[4] = (int2)(1,0);
	corners[5] = (int2)(1,1);
	corners[6] = (int2)(0,1);
	corners[7] = (int2)(-1,1);

	float4 v = vertices[c];
	float zrt = fabs(v.z * fov) * maxFaceDist;
	float4 normal = (float4)(0,0,0,0);
	int count = 0;

	if (v.z != 0.0f) {
		int i=0;
		int2 c1 = coords + corners[i];
		float4 v1 = vertices[c1.y*dim.x+c1.x];

		for (; i<8; i++) {
			int2 c2 = coords + corners[(i+1) % 8];
			float4 v2 = vertices[c2.y*dim.x+c2.x];
			if (length(v-v1) < zrt && length(v-v2) < zrt) {
				normal += cross(v1-v, v2-v);
				count++;
			}
			c1 = c2;
			v1 = v2;
		}
	}

	if (count > 0)
		normals[c] = normalize(normal / count);
	else
		normals[c] = normal;
}

__kernel void mapTexCoords(__global float2* table, __global float2* texCoords, float width, float height) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	int i = coords.y * dim.x + coords.x;
	float2 t = table[i];
	texCoords[i] = (float2)(width * t.x / width, height * t.y / height);
}

__kernel void orthTexCoords(__global float4* vertices, __global float2* texCoords, float x, float y, float scaleX, float scaleY) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int2 dim = (int2)(get_global_size(0), get_global_size(1));
	int i = coords.y * dim.x + coords.x;

	texCoords[i] = vertices[i].xy * (float2)(scaleX,scaleY) + (float2)(x, y);
}

);

//////////////////////////////////////////////////

OpenCLProgramPtr ofxDepthPoints::program;

void ofxDepthPoints::allocate(int numVertices) {
	ofxDepth.setup();
	ofxDepthBuffer::allocate(numVertices);
	vbo.setVertexBuffer(getGLBuffer(), getNumTypePerElement(), getBytesPerElement());
}

void ofxDepthPoints::load(string filepath) {
	ofFile file(filepath);
	string line;
	int width = 0;
	int height = 0;
	int points = 0;
	while(std::getline(file, line) && line.find("DATA") != 0) {
		vector<string> parts = ofSplitString(line, " ");
		if (parts[0] == "WIDTH")
			width = ofToInt(parts[1]);
		if (parts[0] == "HEIGHT")
			height = ofToInt(parts[1]);
		if (parts[0] == "POINTS")
			points = ofToInt(parts[1]);
		cout << line << endl;
	}
	if (line.find("DATA") == 0) {
		size_t pos = file.tellg();
		uint64_t s = file.getSize();
		uint64_t n = (s - pos) / sizeof(float);
		vector<ofVec3f> dat;
		dat.resize(points);
	}
}

void ofxDepthPoints::read(ofxDepthData & data) {
	read(data.getData());
	data.setCount(getNumElements());
}

void ofxDepthPoints::read(vector<ofVec4f> & points) {
	points.resize(getNumElements());
	clBuf.read(points.data(), 0, points.size() * sizeof(ofVec4f));
}

void ofxDepthPoints::write(ofxDepthData & data) {
	write(data.getData(), data.getCount());
}

void ofxDepthPoints::write(vector<ofVec4f> & points, int count) {
	clBuf.write(points.data(), 0, count * sizeof(ofVec4f));
}

void ofxDepthPoints::updateMesh(int width, int height, float noiseThreshold, bool mode) {

	if (!indBuf.isAllocated()) {
		indBuf.allocate(width * height * 6);
		vbo.setIndexBuffer(indBuf.getGLBuffer());
	}
	if (!norBuf.isAllocated()) {
		norBuf.allocate(getNumElements());
		vbo.setNormalBuffer(norBuf.getGLBuffer(), norBuf.getBytesPerElement());
	}

	OpenCLKernelPtr kernel = getKernel("pointsToIndices");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, indBuf.getCLBuffer());
	kernel->setArg(2, norBuf.getCLBuffer());
	kernel->setArg(3, noiseThreshold);
	kernel->setArg(4, tanf(60 * DEG_TO_RAD)/height);
	kernel->run2D(width, height);

	if (mode) {
		kernel = getKernel("calcNormals");
		kernel->setArg(0, getCLBuffer());
		kernel->setArg(1, norBuf.getCLBuffer());
		kernel->setArg(2, noiseThreshold);
		kernel->setArg(3, tanf(60 * DEG_TO_RAD)/height);
		kernel->run2D(width, height);
	}

	vbo.enableIndices();
	vbo.enableNormals();
}

void ofxDepthPoints::updateTexCoords(ofxDepthTable & table, float u, float v) {

	if (!texBuf.isAllocated()) {
		texBuf.allocate(getNumElements());
		vbo.setTexCoordBuffer(texBuf.getGLBuffer(), texBuf.getBytesPerElement());
	}

	OpenCLKernelPtr kernel = getKernel("mapTexCoords");
	kernel->setArg(0, table.getCLBuffer());
	kernel->setArg(1, texBuf.getCLBuffer());
	kernel->setArg(2, u);
	kernel->setArg(3, v);
	kernel->run2D(table.getWidth(), table.getHeight());

	vbo.enableTexCoords();
}

void ofxDepthPoints::updateTexCoords(float x, float y, float scaleX, float scaleY) {

	if (!texBuf.isAllocated()) {
		texBuf.allocate(getNumElements());
		vbo.setTexCoordBuffer(texBuf.getGLBuffer(), texBuf.getBytesPerElement());
	}

	OpenCLKernelPtr kernel = getKernel("orthTexCoords");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, texBuf.getCLBuffer());
	kernel->setArg(2, x);
	kernel->setArg(3, y);
	kernel->setArg(4, scaleX);
	kernel->setArg(5, scaleY);
	kernel->run1D(getNumElements());

	vbo.enableTexCoords();
}

void ofxDepthPoints::draw() {
	vbo.disableIndices();
	vbo.draw(GL_POINTS, 0, getNumElements());
}

void ofxDepthPoints::drawMesh() {
	if (indBuf.isAllocated()) {
		vbo.enableIndices();
		vbo.drawElements(GL_TRIANGLES, indBuf.getNumElements());
	}
}

void ofxDepthPoints::transform(const ofMatrix4x4 &mat, ofxDepthPoints &outputPoints) {

	if (matrix.size() == 0) {
		matrix.initBuffer(4);
	}

	for (int i=0; i<4; i++) {
		matrix[i] = mat.getRowAsVec4f(i);
	}
	matrix.writeToDevice();

	OpenCLKernelPtr kernel = getKernel("transform");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, outputPoints.getCLBuffer());
	kernel->setArg(2, matrix);
	kernel->run1D(getNumElements());
}

void ofxDepthPoints::smoothNormals(int width, int height) {

	if (!norBufTemp.isAllocated()) {
		norBufTemp.allocate(norBuf.getNumElements());
	}
	norBuf.copy(norBufTemp);

	OpenCLKernelPtr kernel = getKernel("smoothNormals");
	kernel->setArg(0, norBufTemp.getCLBuffer());
	kernel->setArg(1, norBuf.getCLBuffer());
	kernel->run2D(width, height);
}

void ofxDepthPoints::transform(const ofMatrix4x4 &mat) {
	transform(mat, *this);
}

ofMesh ofxDepthPoints::makeFrustum(float fovH, float fovV, float clipNear, float clipFar) {
	ofMesh mesh;
	mesh.setMode(OF_PRIMITIVE_LINES);

	float tanH = tan(0.5 * fovH * DEG_TO_RAD);
	float tanV = tan(0.5 * fovV * DEG_TO_RAD);

	ofVec3f cornerFar;
	cornerFar.x = tanH * clipFar;
	cornerFar.y = tanV * clipFar;
	cornerFar.z = -clipFar;

	mesh.addVertex(cornerFar);
	mesh.addVertex(cornerFar * ofVec3f(-1, 1, 1));
	mesh.addVertex(cornerFar * ofVec3f(-1, -1, 1));
	mesh.addVertex(cornerFar * ofVec3f(1, -1, 1));

	ofVec3f cornerNear;
	cornerNear.x = tanH * clipNear;
	cornerNear.y = tanV * clipNear;
	cornerNear.z = -clipNear;

	mesh.addVertex(cornerNear);
	mesh.addVertex(cornerNear * ofVec3f(-1, 1, 1));
	mesh.addVertex(cornerNear * ofVec3f(-1, -1, 1));
	mesh.addVertex(cornerNear * ofVec3f(1, -1, 1));

	mesh.addVertex(ofVec3f(0.f));

	unsigned int indices[] = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 8, 0, 8, 1, 8, 2, 8, 3};
	mesh.addIndices(indices, sizeof(indices)/sizeof(unsigned int));

	return mesh;
}

OpenCLKernelPtr ofxDepthPoints::getKernel(string name) {
	getProgram();
	return ofxDepth.getKernel(name);
}

OpenCLProgramPtr ofxDepthPoints::getProgram() {
	if (program)
		return program;
	else {
		program = ofxDepth.loadProgram(depthPointsProgram);
		ofxDepth.loadKernel("pointsToIndices", program);
		ofxDepth.loadKernel("smoothNormals", program);
		ofxDepth.loadKernel("calcNormals", program);
		ofxDepth.loadKernel("transform", program);
		ofxDepth.loadKernel("mapTexCoords", program);
		ofxDepth.loadKernel("orthTexCoords", program);
		return program;
	}
}

//////////////////////////////////////////////////

void ofxDepthData::allocate(int size) {
	data.resize(size);
}

int ofxDepthData::countZeros() {
	int zeroCount = 0;
	for (ofVec4f & p : data) {
		if (p.x == 0 && p.y == 0 && p.z == 0)
			zeroCount++;
	}
	return zeroCount;
}

int ofxDepthData::removeZeros(bool resize) {
	return removeZeros(resize, *this);
}

int ofxDepthData::removeZeros(bool resize, ofxDepthData &outputData) {
	ofVec4f * inData = getData().data();
	outputData.allocate(getSize());
	ofVec4f * outData = outputData.getData().data();
	int nz = 0;
	size_t size = data.size();
	for (int i=0; i<size; i++) {
		if (inData->x != 0 || inData->y != 0 || inData->z != 0) {
			*outData = *inData;
			outData++;
			nz++;
		}
		inData++;
	}
	outputData.setCount(nz);
	if (resize)
		outputData.allocate(nz);
	return nz;
}