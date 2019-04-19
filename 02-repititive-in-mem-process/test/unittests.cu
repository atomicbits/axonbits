//
// Created by Peter Rigole on 2019-04-19.
//

#include <cuda_runtime.h>

#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#include "util/ArrayTest.cu"


__global__
void run_test(ArrayTest *arrayTest) {
    arrayTest->test();
}

int launch_test(ArrayTest *arrayTest) {
    run_test << < 1, 1 >> > (arrayTest);
    cudaDeviceSynchronize();

    int result = arrayTest->getResult();
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
