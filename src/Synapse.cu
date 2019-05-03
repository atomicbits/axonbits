//
// Created by Peter Rigole on 2019-03-13.
//

#include "Synapse.cuh"

// Default Constructor
Synapse::Synapse() : weight(0.5) {}

Synapse::Synapse(float weight_init, Neuron* source_init) : weight(weight_init), source(source_init) {}

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

__host__ __device__
Neuron *Synapse::getSource() const { return source; }

__host__
void Synapse::setSource(Neuron* neuron) {
    source = neuron;
}
