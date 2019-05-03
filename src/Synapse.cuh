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
    Synapse(float weight_init, Neuron* source_init);

    // Destructor
    ~Synapse();

    // Get the weight
    __host__ __device__
    float getWeight() const;

    // Update the weight
    __host__ __device__
    void updateWeight(const float weight_update);

    // Get the source
    __host__ __device__
    Neuron* getSource() const;

    // Set the source
    __host__
    void setSource(Neuron* neuron);

private:
    // The synapse weight
    float weight;
    // The source neuron
    Neuron* source; // get the source activity via this source neuron
};


#endif //AXONBITS_SYNAPSE_H
