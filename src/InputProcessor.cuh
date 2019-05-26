//
// Created by Peter Rigole on 2019-05-03.
//

#ifndef AXONBITS_INPUTPROCESSOR_H
#define AXONBITS_INPUTPROCESSOR_H

#include "NeuralNet.cuh"
#include "Managed.cuh"

class NeuralNet; // forward declaration to cope with cyclic dependency

/**
 * Input processor instances are only instantiated and used on the host, never on the device, so we can use
 * virtual functions here if we want!
 */
class InputProcessor : public Managed {

public:

    InputProcessor();

    InputProcessor(NeuralNet* neuralNet_init);

    void setNeuralNet(NeuralNet* neuralNet_update);

    virtual void processInput();

private:

    NeuralNet* neuralNet;

};


#endif //AXONBITS_INPUTPROCESSOR_H
