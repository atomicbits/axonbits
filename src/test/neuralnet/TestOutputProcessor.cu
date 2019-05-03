//
// Created by Peter Rigole on 2019-05-03.
//

#ifndef AXONBITS_TESTOUTPUTPROCESSOR_H
#define AXONBITS_TESTOUTPUTPROCESSOR_H

#include "../../OutputProcessor.cuh"
#include "../../NeuralNet.cuh"

class TestOutputProcessor : public OutputProcessor {

public:

    TestOutputProcessor(NeuralNet* neuralNet_init) : OutputProcessor(neuralNet_init) {}

    void processOutput() {

    }

};

#endif //AXONBITS_TESTOUTPUTPROCESSOR_H
