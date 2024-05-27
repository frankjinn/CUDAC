#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define DIM 1000

struct complexNum {
    float r;
    float i;
    __device__ complexNum(float a, float b) : r(a), i(b) {}
    __device__ complexNum operator*(const complexNum& a) {
        return complexNum(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ complexNum operator+(const complexNum& a) {
        return complexNum(r+a.r, i+a.i);
    }
    __device__ float magnitude(void) {
        return r*r + i*i;
    }
};

__device__ int julia(int x, int y) {
    const float scale = 1.5;

    //Scale to -1 to 1, x is real part, y is imaginary part
    float jx = scale*(float) (DIM/2 - x)/(DIM/2);
    float jy = scale*(float) (DIM/2 - y)/(DIM/2);
    
    complexNum c(-0.8, 0.153);
    complexNum a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++){
        a = a* a + c;
        if (a.magnitude() > 1000){
            return 0;
        }
    }

    return 1;
    }

    __global__ void kernel(unsigned char *ptr){
        int x = blockIdx.x;
        int y = blockIdx.y;
        int offset = x + y * gridDim.x;

        int juliaValue = julia(x, y);
        ptr[offset*4 + 0] = 255 * juliaValue;
        ptr[offset*4 + 1] = 0;
        ptr[offset*4 + 2] = 0;
        ptr[offset*4 + 3] = 255;
    }

    int main(void){
        CPUBitmap bitmap(DIM, DIM);
        unsigned char *dev_bitmap;
        HANDLE_ERROR(cudaMalloc((void**) &dev_bitmap, bitmap.image_size()));

        dim3 grid(DIM, DIM);
        kernel<<<grid,1>>>(dev_bitmap);

        HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

        bitmap.display_and_exit();

        HANDLE_ERROR(cudaFree(dev_bitmap));
    }