//
// Created by Peter Rigole on 2019-03-13.
//

#include "Neuron.cuh"

Neuron::Neuron(unsigned long int neuronId,
               NeuronProperties *neuronProperties,
               unsigned int max_nb_incoming_excitatory_synapses,
               unsigned int max_nb_incoming_inhibitory_synapses) :
               id(neuronId),
               properties(neuronProperties) {
    incoming_excitatory_synapses = new Array<Synapse>(max_nb_incoming_excitatory_synapses);
    incoming_inhibitory_synapses = new Array<Synapse>(max_nb_incoming_inhibitory_synapses);
}

Neuron::~Neuron() {
    delete incoming_excitatory_synapses;
    delete incoming_inhibitory_synapses;
    // Pointers that we don't want to delete here are:
    // * properties (because neuron properties are shared)

}

// Get the id
__host__ __device__
unsigned long int Neuron::getId() const { return id; }

__host__ __device__
const NeuronProperties *Neuron::getProperties() const { return properties; }

__host__ __device__
float Neuron::getActivity(CycleParity parity) const {
    if(parity == EvenCycle) return activity_even_parity;
    else return activity_odd_parity;
}

__host__ __device__
void Neuron::updateActivity(float activity_update, CycleParity parity) {
    if(parity == EvenCycle) {
        activity_odd_parity = activity_update;
    } else {
        activity_even_parity = activity_update;
    }

    // ToDo: update long_time_avg_activity
}

__host__
void Neuron::setExternalActivity(float activity_update) {
    activity_even_parity = activity_update;
    activity_odd_parity = activity_update;
}

__host__ __device__
float Neuron::getLongTimeAverageActivity() const { return long_time_avg_activity; }

Array<Synapse>* Neuron::getIncomingExcitatorySynapses() const {
    return incoming_excitatory_synapses;
}

__host__ __device__
void Neuron::addIncomingExcitatorySynapse(Synapse* synapse) {
    incoming_excitatory_synapses->append(synapse);
}

__host__ __device__
Array<Synapse>* Neuron::getIncomingInhibitorySynapses() const {
    return incoming_inhibitory_synapses;
}

__host__ __device__
void Neuron::addIncomingInhibitorySynapse(Synapse* synapse) {
    incoming_inhibitory_synapses->append(synapse);
}
