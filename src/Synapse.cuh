//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef AXONBITS_SYNAPSE_H
#define AXONBITS_SYNAPSE_H

#include "Managed.cuh"
#include "Neuron.cuh"

class Neuron; // forward declaration to cope with cyclic dependency

class Synapse : public Managed {
public:
    // Default Constructor
    Synapse();

    __host__
    Synapse(float weight_init, unsigned int sourceNeuronIndexInit);

    // Copy constructor
    __host__ __device__
    Synapse(const Synapse &synapseOrig);

    // Destructor
    __host__
    ~Synapse();

    // Get the weight
    __host__ __device__
    float getWeight() const;

    // Update the weight
    __host__ __device__
    void updateWeight(const float weight_update);

    // Get the source
    __host__ __device__
    unsigned int getSource() const;

    // Set the source
    __host__
    void setSource(unsigned int sourceNeuronIndexUpdate);

private:
    // The synapse weight
    float weight;
    // The source neuron
    unsigned int sourceNeuronIndex; // get the source activity via this source neuron
};


#endif //AXONBITS_SYNAPSE_H
