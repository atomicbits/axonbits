//
// Created by Peter Rigole on 2019-05-03.
//

#ifndef AXONBITS_NEURANETTEST_H
#define AXONBITS_NEURANETTEST_H

#include <assert.h>
#include "../Test.cuh"

class NeuraNetTest : public Test {

public:

    NeuraNetTest() : Test(TestClass::neuralnettest) {}

    ~NeuraNetTest() {}

    __host__
    void hostTest() {}

    __device__
    void deviceTest() {}

    __host__
    const char* getName() {
        return "NeuralNetTest";
    }

};


#endif //AXONBITS_NEURANETTEST_H
