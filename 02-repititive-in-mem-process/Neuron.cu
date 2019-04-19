//
// Created by Peter Rigole on 2019-03-13.
//

#include "Neuron.cuh"

Neuron::Neuron() : id(0),
                   activity(0),
                   previous_activity(0),
                   long_time_avg_activity(0) {}

// Copy constructor
Neuron::Neuron(const Neuron &neuron) : id(neuron.id),
                                       activity(neuron.activity),
                                       previous_activity(neuron.previous_activity),
                                       long_time_avg_activity(neuron.long_time_avg_activity),
                                       properties(neuron.properties) {}

Neuron::~Neuron() {
    // cudaFree(data); // free the pointers!
}

Neuron::Neuron(unsigned long int neuronId, NeuronProperties *neuronProperties, unsigned int max_nb_incoming_synapses) :
        id(neuronId),
        properties(neuronProperties),
        max_nb_synapses(max_nb_incoming_synapses) {
    incoming_synapses = new Synapse[max_nb_incoming_synapses];
}

// Get the id
__host__ __device__
unsigned long int Neuron::getId() const { return id; }

__host__ __device__
float Neuron::getActivity() const { return activity; }

__host__ __device__
float Neuron::getPreviousActivity() const { return previous_activity; }

__host__ __device__
float Neuron::getLongTimeAverageActivity() const { return long_time_avg_activity; }

__host__ __device__
NeuronProperties *Neuron::getProperties() const { return properties; }
