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
void run_device_test(ArrayTest* test) {
    test->deviceTest();
    return;
}

void run_host_test(ArrayTest* test) {
    test->hostTest();
    return;
}

void checkCudaErrors(const char* testName) {
    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Device failure, cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        return;
    }
    printf("%s device test successful\n", testName);
}

void launch_test(ArrayTest* test) {

    run_device_test<<< 1, 1 >>>(test);
    cudaDeviceSynchronize();
    checkCudaErrors(test->getName());

    run_host_test(test);
    printf("%s host test successful\n", test->getName());

    return;
}


int main(int argc, char **argv) {

    // A sad thing that we can't use polymorphism through virtual function calls when using unified memory!
    // Some ideas for workarounds: https://www.codeproject.com/Articles/603818/Cplusplus-Runtime-Polymorphism-without-Virtual-Fun
    ArrayTest* arrayTest = new ArrayTest();
    launch_test(arrayTest);
    delete arrayTest;

    printf("Testing done.\n");

    cudaDeviceReset();

    return 0;
}
