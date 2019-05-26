//
// Created by Peter Rigole on 2019-05-03.
//

#ifndef AXONBITS_OUTPUTPROCESSOR_H
#define AXONBITS_OUTPUTPROCESSOR_H

#include "NeuralNet.cuh"
#include "Managed.cuh"

class NeuralNet; // forward declaration to cope with cyclic dependency

/**
 * Output processor instances are only instantiated and used on the host, never on the device, so we can use
 * virtual functions here if we want!
 */
class OutputProcessor : public Managed {

public:

    OutputProcessor();

    OutputProcessor(NeuralNet* neuralNet_init);

    void setNeuralNet(NeuralNet* neuralNet_update);

    virtual void processOutput();

private:

    NeuralNet* neuralNet;

};


#endif //AXONBITS_OUTPUTPROCESSOR_H
