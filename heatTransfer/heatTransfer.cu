/*
Attempted to use texture memory to speedup simulation. However, wasn't able to figure it out. Try to do later.
*/
#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_anim.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <texture_types.h>

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.00001f
#define SPEED   0.25f
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
    CPUAnimBitmap  *bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

 __global__ void blend_kernel( float* outSrc, const float* inSrc) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)   left++;
    if (x == DIM-1) right--; 

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0)   top += DIM;
    if (y == DIM-1) bottom -= DIM;

    outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - 4*inSrc[offset]);
 }

 __global__ void copyConstKernel(float* iptr, const float *cptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (cptr[offset] != 0) iptr[offset] = cptr[offset];
 }
 void anim_gpu( DataBlock *d, int ticks ) {
    cudaEventRecord( d->start, 0 );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);
    CPUAnimBitmap  *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;
    for (int i=0; i<360; i++) {
        copyConstKernel<<<blocks,threads>>>( d->dev_inSrc, d->dev_constSrc);
        blend_kernel<<<blocks,threads>>>( d->dev_outSrc, d->dev_inSrc );
        swap(d->dev_inSrc, d->dev_outSrc);
    }
    float_to_color<<<blocks,threads>>>( d->output_bitmap,
                                        d->dev_inSrc );

    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                              d->output_bitmap,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame:  %3.1f ms\n",
            d->totalTime/d->frames  );
}

void anim_exit( DataBlock *d ) {
    cudaFree( d->dev_inSrc );
    cudaFree( d->dev_outSrc );
    cudaFree( d->dev_constSrc);

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}

int main(void) {
    DataBlock   data;
    CPUAnimBitmap bitmap( DIM, DIM, &data );
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );

    int imageSize = bitmap.image_size();

    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
                               imageSize ) );

    // assume float == 4 chars in size (ie rgba)
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc,
                              imageSize ) );

    // initialize the constant data
    float *temp = (float*)malloc( imageSize );
    for (int i=0; i<DIM*DIM; i++) {
        temp[i] = 0;
    }


    for (int y=825; y<875; y++) {
        for (int x=425; x<475; x++) {
            temp[x+y*DIM] = MAX_TEMP*0.5;
        }
    }

    for (int y=475; y<625; y++) {
        for (int x=375; x<525; x++) {
            temp[x+y*DIM] = MIN_TEMP;
        }
    }

    for (int y=545; y<555; y++) {
        for (int x=240; x<250; x++) {
            temp[x+y*DIM] = MAX_TEMP*0.6;
        }
        for (int x=640; x<650; x++) {
            temp[x+y*DIM] = MAX_TEMP*0.6;
        }
    }

    for (int y=225; y<275; y++) {
        for (int x=425; x<475; x++) {
            temp[x+y*DIM] = MAX_TEMP*0.5;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );    

    // initialize the input data
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            temp[x+y*DIM] = 0;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );
    free( temp );

    bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu,
                           (void (*)(void*))anim_exit );
}