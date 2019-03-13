//
// Created by Peter Rigole on 2019-03-13.
//
#include "Managed.cuh"

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H

class NeuralNet : public Managed {
public:
    // Default Constructor
    NeuralNet();

    // Constructor
    NeuralNet(const char*);

    // Copy constructor
    NeuralNet(const NeuralNet&);

    // Destructor
    ~NeuralNet();

    // Get the name
    __host__ __device__
    char* name() const;

private:
    int length;
    char* data;
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURALNET_H
