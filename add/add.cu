#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <stdio.h>
#include "../common/book.h"
using namespace std;

__global__ void add(int a, int b, int* c){
    *c = a + b;
}

int main(void){
    int c;
    int* devPtr;
    HANDLE_ERROR(cudaMalloc(&devPtr, sizeof(int)));
    add<<<1,1>>>(2, 7, devPtr);
    HANDLE_ERROR(cudaMemcpy(
        &c,
        devPtr,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    printf("2 + 7 = %d\n", c);
    cudaFree(devPtr);

    int count;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&count);

    cout<<count<<"\n";

    for (int i = 0; i < count; i++){
        cudaGetDeviceProperties(&prop, i);
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap:  " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( "   --- Memory Information for device %d ---\n", i );
        // printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total global mem (GB):  %zu\n", prop.totalGlobalMem/1000000000 );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }
}