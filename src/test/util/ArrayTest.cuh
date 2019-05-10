//
// Created by Peter Rigole on 2019-05-10.
//

#ifndef AXONBITS_ARRAYTEST_H
#define AXONBITS_ARRAYTEST_H

#include <cuda_runtime.h>
#include <assert.h>
#include "../../util/Array.cu"
#include "../../Managed.cuh"

#include "../Test.cu"
#include "TestContainer.cu"


class ArrayTest : public Managed, public Test {

public:

    // inline static const string TYPE = "ArrayTest";

    __host__
    ArrayTest();

    __host__
    ~ArrayTest();

    __host__
    void test();

    __device__
    void deviceTest();

    __host__
    void hostTest();


private:
    Array<TestContainer>* arr;

};

#endif //AXONBITS_ARRAYTEST_H
