//
// Created by Peter Rigole on 2019-04-24.
//

#ifndef AXONBITS_TEST_H
#define AXONBITS_TEST_H

#include <cuda_runtime.h>
#include <string>


/**
 * Test base class.
 * Mind that classes with virtual functions can't have a header file.
 */
class Test {

public:

    Test() {}

    Test(const char* nameInit) : name(nameInit) {}

    ~Test() {}

    __host__
    virtual void test() {}

    __host__
    void checkCudaErrors() {
        cudaDeviceSynchronize();
        cudaError_t cudaError;
        cudaError = cudaGetLastError();
        if(cudaError != cudaSuccess) {
            printf("%s device failure, cudaGetLastError() returned %d: %s\n", getName(), cudaError, cudaGetErrorString(cudaError));
        } else {
            printf("%s device test successful\n", getName());
        }
    }

    __host__
    const char* getName() { return name; }

private:
    const char* name;

};


#endif //AXONBITS_TEST_H
