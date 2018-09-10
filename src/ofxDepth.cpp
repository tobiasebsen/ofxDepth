#include "ofxDepth.h"

#define STRINGIFY(A) #A

string depthProgram = STRINGIFY(

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

__kernel void pointsFromFov(__global unsigned short* depth, float2 fov, __global float4* points) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    int2 dims = (int2)(get_global_size(0), get_global_size(1));
    float2 angle = fov * (((float2)coords / dims) - float2(0.5f));
    int i = coords.y * dims.x + coords.x;
    points[i].x = tan(radians(angle.x)) * depth[i];
    points[i].y = tan(radians(angle.y)) * depth[i];
    points[i].z = -depth[i];
    points[i].w = 0;
}
                                
__kernel void pointsFromTable(__global unsigned short* depth, __global float2* table, __global float4* points) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    int i = coords.y * get_global_size(0) + coords.x;
    unsigned short d = depth[i];
    points[i].x = table[i].x * d;
    points[i].y = table[i].y * d;
    points[i].z = -d;
    points[i].w = 0.f;
};

__kernel void transform(__global float4 *input, __global float4 *output, __global float4 *mat) {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    int width = get_global_size(0);
    int i = coords.y * width + coords.x;
    float4 v = input[i];
    
    output[i].x = mat[0].x * v.x + mat[1].x * v.y + mat[2].x * v.z + mat[3].x;
    output[i].y = mat[0].y * v.x + mat[1].y * v.y + mat[2].y * v.z + mat[3].y;
    output[i].z = mat[0].z * v.x + mat[1].z * v.y + mat[2].z * v.z + mat[3].z;
}
                                
);

void ofxDepthImage::allocate(int width, int height) {
    this->width = width;
    this->height = height;
    glBuf.allocate(width * height * sizeof(unsigned short), GL_STREAM_DRAW);
    clBuf.initFromGLObject(glBuf.getId());
}

bool ofxDepthImage::isAllocated() const {
    return glBuf.isAllocated();
}

void ofxDepthImage::load(string filepath) {
    ofxDepth.setup();
    ofShortPixels pixels;
    ofLoadImage(pixels, filepath);
    ofShortPixels spixels = pixels.getChannel(0);
    write(spixels);
}

void ofxDepthImage::write(ofShortPixels & pixels) {
    if (!isAllocated())
        allocate(pixels.getWidth(), pixels.getHeight());
    clBuf.write(pixels.getData(), 0, pixels.getWidth() * pixels.getHeight() * pixels.getBytesPerPixel());
}

void ofxDepthImage::read(ofShortPixels & pixels) {
    clBuf.read(pixels.getData(), 0, pixels.getWidth() * pixels.getHeight() * pixels.getBytesPerPixel());
}

void ofxDepthImage::update(ofTexture & tex) {
    glBuf.bind(GL_PIXEL_UNPACK_BUFFER);
    tex.loadData((unsigned short*)NULL, width, height, GL_LUMINANCE);
    glBuf.unbind(GL_PIXEL_UNPACK_BUFFER);
}

void ofxDepthImage::update() {
    update(tex);    
}

void ofxDepthImage::draw(float x, float y) {
    tex.draw(x, y);
}

void ofxDepthImage::draw(float x, float y, float w, float h) {
    tex.draw(x, y, w, h);
}

void ofxDepthImage::denoise(float threshold, int neighbours, ofxDepthImage &outputImage) {
    OpenCLKernelPtr kernel = ofxDepth.getKernel("denoise");
    kernel->setArg(0, getCLBuffer());
    kernel->setArg(1, outputImage.getCLBuffer());
    kernel->setArg(2, threshold);
    kernel->setArg(3, neighbours);
    kernel->run2D(width, height);
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

void ofxDepthImage::map(uint16_t inputMin, uint16_t inputMax, uint16_t outputMin, uint16_t outputMax) {
    map(inputMin, inputMax,  outputMin, outputMax, *this);
}

void ofxDepthImage::toPoints(float fovH, float fovV, ofxDepthPoints & points) {
    
    if (!points.isAllocated())
        points.allocate(getNumPixels());

    OpenCLKernelPtr kernel = ofxDepth.getKernel("pointsFromFov");
    kernel->setArg(0, getCLBuffer());
    kernel->setArg(1, ofVec2f(fovH, fovV));
    kernel->setArg(2, points.getCLBuffer());
    kernel->run2D(getWidth(), getHeight());
    
    points.setCount(getNumPixels());
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
    removeZeros(resize, *this);
}

int ofxDepthData::removeZeros(bool resize, ofxDepthData &outputData) {
    outputData.allocate(getSize());
    ofVec4f * pData = outputData.getData().data();
    int nz = 0;
    for (ofVec4f & p : data) {
        if (p.x != 0 || p.y != 0 || p.z != 0) {
            data[nz] = p;
            nz++;
        }
    }
    outputData.setCount(nz);
    if (resize)
        outputData.allocate(nz);
    return nz;
}

//////////////////////////////////////////////////

void ofxDepthPoints::allocate(int size) {
    this->size = 0;
    glBuf.allocate(size * sizeof(ofVec4f), GL_STREAM_DRAW);
    vbo.setVertexBuffer(glBuf, 3, sizeof(ofVec4f));
    clBuf.initFromGLObject(glBuf.getId());
}

bool ofxDepthPoints::isAllocated() {
    return glBuf.isAllocated();
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
    data.setCount(getCount());
}

void ofxDepthPoints::read(vector<ofVec4f> & points) {
    points.resize(getCount());
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
    this->count = count;
}

void ofxDepthPoints::draw() {
    vbo.draw(GL_POINTS, 0, count);
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
    kernel->run1D(count);
}

void ofxDepthPoints::transform(const ofMatrix4x4 &mat) {
    transform(mat, *this);
}

//////////////////////////////////////////////////

void ofxDepthCore::setup(int deviceNumber) {

    if (isSetup())
        return;

	opencl.setupFromOpenGL(deviceNumber);
	opencl.loadProgramFromSource(depthProgram);
	opencl.loadKernel("map");
    opencl.loadKernel("pointsFromFov");
    opencl.loadKernel("pointsFromTable");
    opencl.loadKernel("transform");
    opencl.loadKernel("denoise");
}

void ofxDepthCore::setup(string vendorName, string deviceName) {
    int n = opencl.getDeviceInfos();
    for (int i=0; i<n; i++) {
        if ((string((char*)opencl.deviceInfo[i].vendorName) == vendorName && deviceName == "") ||
            string((char*)opencl.deviceInfo[i].deviceName) == deviceName) {
            setup(i);
            return;
        }
    }
}

bool ofxDepthCore::isSetup() {
    return opencl.getDevice() != nullptr;
}

OpenCL & ofxDepthCore::getCL() {
    setup();
    return opencl;
}

OpenCLKernelPtr ofxDepthCore::getKernel(string name) {
    return getCL().kernel(name);
}
