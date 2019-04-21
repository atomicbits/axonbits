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
void run_test(ArrayTest *arrayTest) {
    arrayTest->test();
    return;
}

int launch_test(ArrayTest *arrayTest) {

    int result = arrayTest->getResult();
    printf("Result before is %i\n", result);

    run_test<<< 1, 1 >>>(arrayTest);
    cudaDeviceSynchronize();


    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess)
    {
        printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    }


    result = arrayTest->getResult();
    printf("Result after is %i\n", result);

    Array<TestContainer>* arr = arrayTest->getArray();
    assert((*arr)[0]->getC() == 3.0); // should be 3.0!
    assert((*arr)[2]->getZ() == 9); // should be 15!


    if (result == 0)
        printf("ArrayTest Successful\n");
    else
        printf("ArrayTest Failed\n");

    delete arrayTest;

    return result;
}


int main(int argc, char **argv) {

    int result = launch_test(new ArrayTest());

    printf("Testing done.\n");

    cudaDeviceReset();

    return result;
}
