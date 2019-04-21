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

int launch_test(ArrayTest *arrayTest) {

    run_device_test<<< 1, 1 >>>(arrayTest);
    cudaDeviceSynchronize();

    int result;
    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Device failure, cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        result = arrayTest->getResult();
        delete arrayTest;
        return result;
    }

    if (arrayTest->getResult() == 0) {
        printf("ArrayTest device test successful\n");
    }
    else {
        printf("ArrayTest device test failed\n");
        result = arrayTest->getResult();
        delete arrayTest;
        return result;
    }

    run_host_test(arrayTest);

    if (arrayTest->getResult() == 0)
        printf("ArrayTest host test successful\n");
    else
        printf("ArrayTest host test failed\n");

    result = arrayTest->getResult();
    delete arrayTest;
    return result;
}


int main(int argc, char **argv) {

    int result = launch_test(new ArrayTest());

    printf("Testing done.\n");

    cudaDeviceReset();

    return result;
}
