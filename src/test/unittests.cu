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
void run_device_test(Test* test) {
    test->deviceTest();
    return;
}

void run_host_test(Test* test) {
    test->hostTest();
    return;
}

void launch_test(Test* test) {

    run_device_test<<< 1, 1 >>>(test);
    cudaDeviceSynchronize();

    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        printf("Device failure, cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        return;
    }
    printf("%s device test successful\n", test->getName());

    run_host_test(test);
    printf("%s host test successful\n", test->getName());

    return;
}


int main(int argc, char **argv) {

    // A sad thing that we can't use polymorphism through virtual function calls when using unified memory!
    // Some ideas for workarounds: https://www.codeproject.com/Articles/603818/Cplusplus-Runtime-Polymorphism-without-Virtual-Fun
    // But they all look nasty. Look how ugly it can get: https://stackoverflow.com/questions/22822836/type-switch-construct-in-c11
    ArrayTest* arrayTest = new ArrayTest();
    launch_test(arrayTest);
    delete arrayTest;

    printf("Testing done.\n");

    cudaDeviceReset();

    return 0;
}
