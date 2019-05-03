//
// Created by Peter Rigole on 2019-05-03.
//

#ifndef AXONBITS_TESTINTPUTPROCESSOR_H
#define AXONBITS_TESTINPUTPROCESSOR_H

#include "../../InputProcessor.cuh"
#include "../../NeuralNet.cuh"

class TestInputProcessor : public InputProcessor {

public:

    TestInputProcessor(NeuralNet* neuralNet_init) : InputProcessor(neuralNet_init) {}

    void processInput() {

    }

};

#endif //AXONBITS_TESTINPUTPROCESSOR_H
