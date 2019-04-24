//
// Created by Peter Rigole on 2019-03-13.
//
#include "NeuralNet.cuh"

NeuralNet::NeuralNet(unsigned int maxNeurons_init)  {
    neurons = new Array<Neuron>(maxNeurons_init);
}

NeuralNet::~NeuralNet() {
    delete neurons;
}

__host__
void NeuralNet::trial() {

}

__host__
void NeuralNet::getActivity(float[], unsigned int fromNeuronId, unsigned int toNeuronId) const {
    return;
}

__host__
void NeuralNet::updateActivity(float[], unsigned int fromNeuronId, unsigned int toNeuronId) {
    return;
}

__host__ __device__
void cycle(const Phase phase) {
    return;
}

__host__ __device__
void updateWeights() {
    return;
}

