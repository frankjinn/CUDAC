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
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

// these exist on the GPU side
// texture<float,2>  texConstSrc;
// texture<float,2>  texIn;
// texture<float,2>  texOut;
 __device__  cudaTextureObject_t texConstSrc;
 __device__  cudaTextureObject_t texIn;
 __device__  cudaTextureObject_t texOut;

__global__ void blend_kernel( float *dst,
                              bool dstOut ) {

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
    float   t, l, c, r, b;
    if (dstOut) {
        t = tex2D<float>(texIn,x,y-1);
        l = tex2D<float>(texIn,x-1,y);
        c = tex2D<float>(texIn,x,y);
        r = tex2D<float>(texIn,x+1,y);
        b = tex2D<float>(texIn,x,y+1);
    } else {
        t = tex2D<float>(texOut,x,y-1);
        l = tex2D<float>(texOut,x-1,y);
        c = tex2D<float>(texOut,x,y);
        r = tex2D<float>(texOut,x+1,y);
        b = tex2D<float>(texOut,x,y+1);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

__global__ void copy_const_kernel( float *iptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D<float>(texConstSrc,x,y);
    if (c != 0)
        iptr[offset] = c;
}

// globals needed by the update routine
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

void anim_gpu( DataBlock *d, int ticks ) {
    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);
    CPUAnimBitmap  *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;
    for (int i=0; i<90; i++) {
        float   *in, *out;
        if (dstOut) {
            in  = d->dev_inSrc;
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
            in  = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks,threads>>>( in );
        blend_kernel<<<blocks,threads>>>( out, dstOut );
        dstOut = !dstOut;
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

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) {
    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_constSrc ) );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


int main( void ) {
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

    // create const texture object
    cudaResourceDesc resDescConst;
    memset(&resDescConst, 0, sizeof(resDescConst));
    resDescConst.resType = cudaResourceTypePitch2D;
    resDescConst.res.pitch2D.devPtr = data.dev_constSrc;
    resDescConst.res.pitch2D.width = DIM;
    resDescConst.res.pitch2D.height = DIM; // bits per channel
    resDescConst.res.pitch2D.pitchInBytes = sizeof(float) * DIM;

    cudaTextureDesc texDescConst;
    memset(&texDescConst, 0, sizeof(texDescConst));

    cudaTextureObject_t tempConstSrc;
    cudaCreateTextureObject(&tempConstSrc, &resDescConst, &texDescConst, NULL);
    cudaMemcpyToSymbol(texConstSrc, &tempConstSrc, sizeof(cudaTextureObject_t));

    // create texIn texture object
    cudaResourceDesc resDescIn;
    memset(&resDescIn, 0, sizeof(resDescIn));
    resDescIn.resType = cudaResourceTypePitch2D;
    resDescIn.res.pitch2D.devPtr = data.dev_constSrc;
    resDescIn.res.pitch2D.width = DIM;
    resDescIn.res.pitch2D.height = DIM; // bits per channel
    resDescIn.res.pitch2D.pitchInBytes = sizeof(float) * DIM;

    cudaTextureDesc texDescIn;
    memset(&texDescIn, 0, sizeof(texDescIn));

    cudaTextureObject_t tempIn;
    cudaCreateTextureObject(&tempIn, &resDescIn, &texDescIn, NULL);
    cudaMemcpyToSymbol(texIn, &tempIn, sizeof(cudaTextureObject_t));

    // create texOut texture object
    cudaResourceDesc resDescOut;
    memset(&resDescOut, 0, sizeof(resDescOut));
    resDescOut.resType = cudaResourceTypePitch2D;
    resDescOut.res.pitch2D.devPtr = data.dev_constSrc;
    resDescOut.res.pitch2D.width = DIM;
    resDescOut.res.pitch2D.height = DIM; // bits per channel
    resDescOut.res.pitch2D.pitchInBytes = sizeof(float) * DIM;

    cudaTextureDesc texDescOut;
    memset(&texDescOut, 0, sizeof(texDescOut));

    cudaTextureObject_t tempOut;
    cudaCreateTextureObject(&tempOut, &resDescOut, &texDescOut, NULL);
    cudaMemcpyToSymbol(texOut, &tempOut, sizeof(cudaTextureObject_t));

    // initialize the constant data
    float *temp = (float*)malloc( imageSize );
    for (int i=0; i<DIM*DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x>300) && (x<600) && (y>310) && (y<601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
    temp[DIM*700+100] = MIN_TEMP;
    temp[DIM*300+300] = MIN_TEMP;
    temp[DIM*200+700] = MIN_TEMP;
    for (int y=800; y<900; y++) {
        for (int x=400; x<500; x++) {
            temp[x+y*DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );    

    // initialize the input data
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            temp[x+y*DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );
    free( temp );

    bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu,
                           (void (*)(void*))anim_exit );
}
