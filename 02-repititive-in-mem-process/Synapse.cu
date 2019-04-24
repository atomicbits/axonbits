//
// Created by Peter Rigole on 2019-03-13.
//

#include "Synapse.cuh"

// Default Constructor
Synapse::Synapse() : weight(0.5),
                     short_time_synaptic_activity(0),
                     medium_time_synaptic_activity(0) {}

// Destructor
Synapse::~Synapse() {

    // Pointers that we don't want to delete here are:
    // * source (because source neurons are managed elsewhere)
}

// Get the weight
__host__ __device__
float Synapse::getWeight() const { return weight; }

__host__ __device__
float Synapse::getShortTimeSynapticActivity() const { return short_time_synaptic_activity; }

__host__ __device__
float Synapse::getMediumTimeSynapicActivity() const { return medium_time_synaptic_activity; }

// Get the source
__host__ __device__
Neuron *Synapse::getSource() const { return source; }
