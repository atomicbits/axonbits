//
// Created by Peter Rigole on 2019-03-13.
//

#include "Synapse.cuh"

// Default Constructor
Synapse::Synapse() : weight(0.5) {}

Synapse::Synapse(float weight_init, unsigned int sourceNeuronIndexInit) :
                    weight(weight_init), sourceNeuronIndex(sourceNeuronIndexInit) {}

__host__ __device__
Synapse::Synapse(const Synapse &synapseOrig) {
    weight = synapseOrig.weight;
    sourceNeuronIndex = synapseOrig.sourceNeuronIndex;
}

// Destructor
__host__
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
unsigned int Synapse::getSource() const { return sourceNeuronIndex; }

__host__
void Synapse::setSource(unsigned int sourceNeuronIndexUpdate) {
    sourceNeuronIndex = sourceNeuronIndexUpdate;
}
