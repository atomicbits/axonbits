//
// Created by Peter Rigole on 2019-03-13.
//

#include "Neuron.cuh"

Neuron::Neuron(unsigned long int neuronId,
               NeuronProperties *neuronProperties,
               unsigned int max_nb_incoming_synapses) :
               id(neuronId),
               properties(neuronProperties) {
    incoming_synapses = new Array<Synapse>(max_nb_incoming_synapses);
}

Neuron::~Neuron() {
    delete incoming_synapses;
    // Pointers that we don't want to delete here are:
    // * properties (because neuron properties are shared)

}

// Get the id
__host__ __device__
unsigned long int Neuron::getId() const { return id; }

__host__ __device__
float Neuron::getActivity() const { return activity; }

__host__ __device__
void Neuron::updateActivity(float activity_update) {
    previous_activity = activity;
    activity = activity_update;

    // ToDo: update long_time_avg_activity
}

__host__ __device__
float Neuron::getPreviousActivity() const { return previous_activity; }

__host__ __device__
float Neuron::getLongTimeAverageActivity() const { return long_time_avg_activity; }

__host__ __device__
const NeuronProperties *Neuron::getProperties() const { return properties; }

Array<Synapse>* Neuron::getIncomingSynapses() const { return incoming_synapses; }

__host__ __device__
void Neuron::addIncomingSynapse(Synapse* synapse) {
    incoming_synapses->append(synapse);
}
