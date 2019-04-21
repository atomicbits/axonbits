//
// Created by Peter Rigole on 2019-04-19.
//

#include <cuda_runtime.h>

#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>

#include "util/ArrayTest.cu"


__global__
void run_device_test(ArrayTest *arrayTest) {
    arrayTest->deviceTest();
    return;
}

void run_host_test(ArrayTest *arrayTest) {
    arrayTest->hostTest();
    return;
}

void launch_test(ArrayTest *arrayTest) {

    run_device_test<<< 1, 1 >>>(arrayTest);
    cudaDeviceSynchronize();

    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Device failure, cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        delete arrayTest;
        return;
    }

    printf("ArrayTest device test successful\n");

    run_host_test(arrayTest);

    printf("ArrayTest host test successful\n");

    delete arrayTest;
    return;
}


int main(int argc, char **argv) {

    launch_test(new ArrayTest());

    printf("Testing done.\n");

    cudaDeviceReset();

    return 0;
}
