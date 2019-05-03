//
// Created by Peter Rigole on 2019-03-13.
//

#include "Synapse.cuh"

// Default Constructor
Synapse::Synapse() : weight(0.5) {}

// Destructor
Synapse::~Synapse() {

    // Pointers that we don't want to delete here are:
    // * source (because source neurons are managed elsewhere)
}

// Get the weight
__host__ __device__
float Synapse::getWeight() const { return weight; }

__host__ __device__
void Synapse::updateWeight(const float weight_update) {
    weight = weight_update;
}

// Get the source
__host__ __device__
Neuron *Synapse::getSource() const { return source; }
