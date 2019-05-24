//
// Created by Peter Rigole on 2019-04-19.
//

#include <cuda_runtime.h>

#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>

#include "util/ArrayTest.cuh"
#include "neuralnet/NeuralNetTest.cu"


int main(int argc, char **argv) {

    // A sad thing that we can't use polymorphism through virtual function calls when using unified memory!
    // Some ideas for workarounds: https://www.codeproject.com/Articles/603818/Cplusplus-Runtime-Polymorphism-without-Virtual-Fun
    // But they all look nasty. Look how ugly it can get: https://stackoverflow.com/questions/22822836/type-switch-construct-in-c11
    ArrayTest* arrayTest = new ArrayTest();
    arrayTest->test();
    delete arrayTest;

    NeuralNetTest* neuralNetTest = new NeuralNetTest();
    neuralNetTest->test();
    // delete neuralNetTest;

    printf("Testing done.\n");

    cudaDeviceReset();

    return 0;
}
