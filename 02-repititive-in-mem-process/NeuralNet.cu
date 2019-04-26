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
    for (int i = 0; i < 75; i++) {
        cycle(ExpectationPhase, getParity(i));
    }
    for (int i = 75; i < 100; i++) {
        cycle(OutcomePhase, getParity(i));
    }
    updateWeights();
}

__host__
void NeuralNet::getActivity(float activity[], unsigned int fromNeuronId, unsigned int toNeuronId) const {
    return;
}

__host__
void NeuralNet::updateActivity(float activity[], unsigned int fromNeuronId, unsigned int toNeuronId) {
    return;
}

__host__
void NeuralNet::cycle(const Phase phase, const CycleParity parity) {

    return;
}

__host__
void NeuralNet::updateWeights() {
    return;
}

__host__
CycleParity NeuralNet::getParity(int i) {
    if(i % 2 == 0) return EvenCycle;
    else return OddCycle;
}
