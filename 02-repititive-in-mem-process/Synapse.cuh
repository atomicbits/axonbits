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

    // Copy constructor
    Synapse(const Synapse&);

    // Destructor
    ~Synapse();

    // Get the weight
    __host__ __device__
    float getWeight() const;

    __host__ __device__
    float getShortTimeSynapticActivity() const;

    __host__ __device__
    float getMediumTimeSynapicActivity() const;

    // Get the source
    __host__ __device__
    Neuron* getSource();

private:
    // The synapse weight
    float weight;
    // x_s * y_s averaged during the last 25ms (during the last quarter of a trial)
    float short_time_synaptic_activity;
    // x_m * y_m averaged during the first 75ms of each trial
    float medium_time_synaptic_activity;
    // The source neuron
    Neuron *source; // get the source activity via this source neuron
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_SYNAPSE_H
