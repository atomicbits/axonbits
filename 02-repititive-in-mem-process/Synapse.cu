//
// Created by Peter Rigole on 2019-03-13.
//

#include "Synapse.cuh"

// Default Constructor
Synapse::Synapse() {}

// Copy constructor
Synapse::Synapse(const Synapse&) {}

// Destructor
Synapse::~Synapse() {}

// Get the weight
__host__ __device__
float Synapse::getWeight() const { return weight; }

// Get the target
__host__ __device__
Neuron* Synapse::getTarget() { return target; }
