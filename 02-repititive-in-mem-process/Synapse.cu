//
// Created by Peter Rigole on 2019-03-13.
//

#include "Synapse.cuh"

// Default Constructor
Synapse::Synapse() : weight(0.5),
                     short_time_synaptic_activity(0),
                     medium_time_synaptic_activity(0) {}

// Copy constructor
Synapse::Synapse(const Synapse &synapse) : weight(synapse.weight),
                                           short_time_synaptic_activity(synapse.short_time_synaptic_activity),
                                           medium_time_synaptic_activity(synapse.medium_time_synaptic_activity),
                                           source(synapse.source) {}

// Destructor
Synapse::~Synapse() {}

// Get the weight
__host__ __device__
float Synapse::getWeight() const { return weight; }

__host__ __device__
float Synapse::getShortTimeSynapticActivity() const { return short_time_synaptic_activity; }

__host__ __device__
float Synapse::getMediumTimeSynapicActivity() const { return medium_time_synaptic_activity; }

// Get the source
__host__ __device__
Neuron *Synapse::getSource() { return source; }
