#include "ofxDepth.h"

#define STRINGIFY(A) #A

string depthProgram = STRINGIFY(

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
    
    int xmin = coords.x >= 1 ? coords.x-1 : 0;
    int xmax = coords.x < width-1 ? coords.x+1 : width-1;
    int ymin = coords.y >= 1 ? coords.y-1 : 0;
    int ymax = coords.y < height-1 ? coords.y+1 : height-1;
    
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
};

__kernel void map(__global unsigned short* depthIn, unsigned short imin, unsigned short imax, unsigned short omin, unsigned short omax, __global unsigned short* depthOut) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
    if (depthIn[i] == 0)
        depthOut[i] = 0;
    else
        depthOut[i] = (( (int)depthIn[i] - imin) * ( (int)omax - omin)) / ( (int)imax - imin) + omin;
}

__kernel void accumulate(__global unsigned short* input, __global unsigned short* output, float amount) {
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
	int i = coords.y * get_global_size(0) + coords.x;
	if (input[i] > 0) {
		if (output[i] == 0)
			output[i] = input[i];
		else
			output[i] = output[i] * (1-amount) + input[i] * amount;
	}
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
    points[i].z = d == 0 ? 10000 : -d; // HACK: Keeps zero-points away from view
    points[i].w = 1.f;
};

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

__kernel void pointsToIndices(__global float4* vertices, __global unsigned int* indices, __global float4* normals, float maxFaceDist) {
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

	if (v1.z != 0.0 && v4.z != 0.0 && v2.z != 0.0 && fabs(v1.z - v2.z) < maxFaceDist && fabs(v1.z - v4.z) < maxFaceDist && coords.x < dim.x-1 && coords.y < dim.y-1) {
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
	if (v2.z != 0.0 && v4.z != 0.0 && v3.z != 0.0 && fabs(v2.z - v3.z) < maxFaceDist && fabs(v3.z - v4.z) < maxFaceDist && coords.x < dim.x-1 && coords.y < dim.y-1) {
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
                                
);


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

//////////////////////////////////////////////////

void ofxDepthTable::allocate(int width, int height) {
	ofxDepthBuffer::allocate(width * height);
}

//////////////////////////////////////////////////

void ofxDepthImage::load(string filepath) {
    ofxDepth.setup();
    ofShortPixels pixels;
    ofLoadImage(pixels, filepath);
    ofShortPixels spixels = pixels.getChannel(0);
    write(spixels);
}

void ofxDepthImage::read(ofShortPixels & pixels) {
	if (!isAllocated())
		return;
	if (!pixels.isAllocated())
		pixels.allocate(getWidth(), getHeight(), 1);
    clBuf.read(pixels.getData(), 0, pixels.getWidth() * pixels.getHeight() * pixels.getBytesPerPixel());
}

void ofxDepthImage::update(ofTexture & tex) {
    glBuf.bind(GL_PIXEL_UNPACK_BUFFER);
    tex.loadData((unsigned short*)NULL, getWidth(), getHeight(), GL_LUMINANCE);
    glBuf.unbind(GL_PIXEL_UNPACK_BUFFER);
}

void ofxDepthImage::update() {
    update(tex);    
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

void ofxDepthImage::map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin, uint16_t outputMax, ofxDepthImage & outpuImage) {
    
    if (!outpuImage.isAllocated())
        outpuImage.allocate(getWidth(), getHeight());
    
    OpenCLKernelPtr kernel = ofxDepth.getKernel("map");
    kernel->setArg(0, getCLBuffer());
    kernel->setArg(1, &inputMin, sizeof(uint16_t));
    kernel->setArg(2, &inputMax, sizeof(uint16_t));
    kernel->setArg(3, &outputMin, sizeof(uint16_t));
    kernel->setArg(4, &outputMax, sizeof(uint16_t));
    kernel->setArg(5, outpuImage.getCLBuffer());
    kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::accumulate(ofxDepthImage & outputImage, float amount) {

	if (!outputImage.isAllocated())
		outputImage.allocate(getWidth(), getHeight());

	OpenCLKernelPtr kernel = ofxDepth.getKernel("accumulate");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, outputImage.getCLBuffer());
	kernel->setArg(2, amount);
	kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::subtract(ofxDepthImage & background, int threshold) {

	if (!background.isAllocated())
		return;

	OpenCLKernelPtr kernel = ofxDepth.getKernel("subtract");
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

    OpenCLKernelPtr kernel = ofxDepth.getKernel("pointsFromFov");
    kernel->setArg(0, getCLBuffer());
    kernel->setArg(1, ofVec2f(fovH, fovV));
    kernel->setArg(2, points.getCLBuffer());
    kernel->run2D(getWidth(), getHeight());
}

void ofxDepthImage::toPoints(ofxDepthTable & table, ofxDepthPoints & points) {

	if (!points.isAllocated())
		points.allocate(getNumElements());

	OpenCLKernelPtr kernel = ofxDepth.getKernel("pointsFromTable");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, table.getCLBuffer());
	kernel->setArg(2, points.getCLBuffer());
	kernel->run2D(getWidth(), getHeight());
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

//////////////////////////////////////////////////

void ofxDepthPoints::allocate(int numVertices) {
	ofxDepth.setup();
    ofxDepthBuffer::allocate(numVertices);
    vbo.setVertexBuffer(glBuf, getNumTypePerElement(), getBytesPerElement());
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

void ofxDepthPoints::write(vector<ofVec4f> & points) {
    write(points, points.size());
}

void ofxDepthPoints::write(vector<ofVec4f> & points, int count) {
    clBuf.write(points.data(), 0, count * sizeof(ofVec4f));
}

void ofxDepthPoints::updateMesh(int width, int height, float noiseThreshold) {

	if (!indBuf.isAllocated()) {
		indBuf.allocate(width * height * 6);
		vbo.setIndexBuffer(indBuf.getGLBuffer());
	}
	if (!norBuf.isAllocated()) {
		norBuf.allocate(getNumElements());
		vbo.setNormalBuffer(norBuf.getGLBuffer(), norBuf.getBytesPerElement());
	}

	OpenCLKernelPtr kernel = ofxDepth.getKernel("pointsToIndices");
	kernel->setArg(0, getCLBuffer());
	kernel->setArg(1, indBuf.getCLBuffer());
	kernel->setArg(2, norBuf.getCLBuffer());
	kernel->setArg(3, noiseThreshold);
	kernel->run2D(width, height);
}

void ofxDepthPoints::draw() {
	vbo.disableIndices();
    vbo.draw(GL_POINTS, 0, getNumElements());
}

void ofxDepthPoints::drawMesh() {
	if (indBuf.isAllocated()) {
		vbo.enableIndices();
		vbo.enableNormals();
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
    
    OpenCLKernelPtr kernel = ofxDepth.getKernel("transform");
    kernel->setArg(0, getCLBuffer());
    kernel->setArg(1, outputPoints.getCLBuffer());
    kernel->setArg(2, matrix);
    kernel->run1D(getNumElements());
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

//////////////////////////////////////////////////

void ofxDepthCore::setup(int deviceNumber) {
    if (isSetup())
        return;
	opencl.setupFromOpenGL(deviceNumber);
	loadKernels();
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

void ofxDepthCore::loadKernels() {
	opencl.loadProgramFromSource(depthProgram);
	opencl.loadKernel("flipH");
	opencl.loadKernel("flipV");
	opencl.loadKernel("limit");
	opencl.loadKernel("map");
	opencl.loadKernel("accumulate");
	opencl.loadKernel("subtract");
	opencl.loadKernel("pointsFromFov");
	opencl.loadKernel("pointsFromTable");
	opencl.loadKernel("transform");
	opencl.loadKernel("denoise");
	opencl.loadKernel("pointsToIndices");
}

OpenCL & ofxDepthCore::getCL() {
    setup();
    return opencl;
}

OpenCLKernelPtr ofxDepthCore::getKernel(string name) {
    return getCL().kernel(name);
}
