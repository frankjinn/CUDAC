#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>
#include "../common/cpu_bitmap.h"
#define INF 2e10f
#define rnd(x) (x*rand()/RAND_MAX)
#define SPHERES 20
#define DIM 1024

struct Sphere {
    float r,g,b;
    float radius;
    float x,y,z;

    __device__ float hit(float rayX, float rayY, float *colourGrad) {
        float dx = rayX - x;
        float dy = rayY - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *colourGrad = dz/sqrtf(radius*radius);
            return dz + z;
        }
        return -INF;
    }
};

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float rayX = x - (DIM/2);
    float rayY = y - (DIM/2);
    
    float r=0, g=0, b=0;
    float maxz = -INF;

    for(int i=0; i<SPHERES; i++){
        float n, t = s[i].hit(rayX, rayY, &n);
        if (t>maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }
    ptr[offset*4 + 0] = (int (r*255));
    ptr[offset*4 + 1] = (int (g*255));
    ptr[offset*4 + 2] = (int (b*255));
    ptr[offset*4 + 3] = 255;
}

// // globals needed by the update routine
// struct DataBlock {
//     unsigned char   *dev_bitmap;
// };

int main(void) {
    // DataBlock   data;
    //Capture program start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *devBitmap;

    cudaMalloc((void**) &devBitmap, bitmap.image_size());

    Sphere *tempS = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i=0; i<SPHERES; i++) {
        tempS[i].r = rnd(1.0f);
        tempS[i].g = rnd(1.0f);
        tempS[i].b = rnd(1.0f);
        tempS[i].x = rnd(1000.0f) - 500;
        tempS[i].y = rnd(1000.0f) - 500;
        tempS[i].z = rnd(1000.0f) - 500;
        tempS[i].radius = rnd(100.0f) + 20;
    }

    cudaMemcpyToSymbol(s, tempS, sizeof(Sphere) * SPHERES);
    free(tempS);

    //Generate bitmap
    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);
    kernel<<<grids, threads>>>(devBitmap);

    cudaMemcpy(bitmap.get_ptr(), devBitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    //Record Endtime
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::printf("Time to generate: %3.1f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    bitmap.display_and_exit();
    cudaFree(devBitmap);
}

