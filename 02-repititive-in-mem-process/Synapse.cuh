//
// Created by Peter Rigole on 2019-03-13.
//

#include "Neuron.cuh"

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_SYNAPSE_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_SYNAPSE_H


class Synapse {
public:
    // Default Constructor
    Synapse();

    // Copy constructor
    Synapse(const Synapse&);

    // Destructor
    ~Synapse();

    // Get the weight
    __host__ __device__
    float getWeight() const;

    // Get the target
    __host__ __device__
    Neuron* getTarget();

private:
    float weight;
    Neuron *target;
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_SYNAPSE_H
