//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_SYNAPSE_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_SYNAPSE_H

#include "Managed.cuh"
#include "Neuron.cuh"

class Neuron; // forward declaration to cope with cyclic dependency

class Synapse : public Managed {
public:
    // Default Constructor
    Synapse();

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

private:
    // The synapse weight
    float weight;
    // The source neuron
    Neuron* source; // get the source activity via this source neuron
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_SYNAPSE_H
