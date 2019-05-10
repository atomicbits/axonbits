//
// Created by Peter Rigole on 2019-03-13.
//

#include "Neuron.cuh"

Neuron::Neuron(NeuronProperties *neuronProperties,
               unsigned int max_nb_incoming_excitatory_synapses,
               unsigned int max_nb_incoming_inhibitory_synapses) :
               properties(neuronProperties),
               short_time_sum_activity(0.0),
               medium_time_sum_activity(0.0),
               long_time_avg_activity(0.0) {
    incoming_excitatory_synapses = new Array<Synapse>(max_nb_incoming_excitatory_synapses);
    incoming_inhibitory_synapses = new Array<Synapse>(max_nb_incoming_inhibitory_synapses);
}

__host__ __device__
Neuron::Neuron(const Neuron &neuronOrig) {
    properties = neuronOrig.properties;
    activity_even_parity = neuronOrig.activity_even_parity;
    activity_odd_parity = neuronOrig.activity_odd_parity;
    short_time_sum_activity = neuronOrig.short_time_sum_activity;
    medium_time_sum_activity = neuronOrig.medium_time_sum_activity;
    long_time_avg_activity = neuronOrig.long_time_avg_activity;
    incoming_excitatory_synapses = neuronOrig.incoming_excitatory_synapses;
    incoming_inhibitory_synapses = neuronOrig.incoming_inhibitory_synapses;
}

__host__
Neuron::~Neuron() {
    delete incoming_excitatory_synapses;
    delete incoming_inhibitory_synapses;
    // Pointers that we don't want to delete here are:
    // * properties (because neuron properties are shared)

}

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
}

__host__
void Neuron::setExternalActivity(float activity_update) {
    activity_even_parity = activity_update;
    activity_odd_parity = activity_update;
}


__host__ __device__
float Neuron::getShortTimeAverageActivity(const int fourthQuarterLength) const {
    return short_time_sum_activity / (float)fourthQuarterLength;
}

__device__
void Neuron::resetShortTimeSumActivity() {
    short_time_sum_activity = 0.0;
}

__device__
void Neuron::incrementShortTimeSumActivity(const float activity) {
    short_time_sum_activity += activity;
}

__host__ __device__
float Neuron::getMediumTimeAverageActivity(const int threeQuarterLength) const {
    return medium_time_sum_activity / (float)threeQuarterLength;
}

__device__
void Neuron::resetMediumTimeSumActivity() {
    medium_time_sum_activity = 0.0;
}

__device__
void Neuron::incrementMediumTimeSumActivity(const float activity) {
    medium_time_sum_activity += activity;
}

__host__ __device__
float Neuron::getLongTimeAverageActivity() const {
    return long_time_avg_activity;
}

/**
 * We only have to increment this activity once per trial.
 * We use the average on the last quarter as an approximation to update the long time average with (not sure yet
 * if this is a good idea).
 */
__device__
void Neuron::incrementLongTimeAverageActivity(const float activity, const float alpha) {
    // y_l = y_l + alpha * (y - y_l)
    // alpha should be closer to 0 to get a long term average
    long_time_avg_activity += alpha * (activity - long_time_avg_activity);
}

Array<Synapse>* Neuron::getIncomingExcitatorySynapses() const {
    return incoming_excitatory_synapses;
}

__host__
void Neuron::addIncomingExcitatorySynapse(Synapse &synapse) {
    incoming_excitatory_synapses->append(synapse);
}

__host__ __device__
Array<Synapse>* Neuron::getIncomingInhibitorySynapses() const {
    return incoming_inhibitory_synapses;
}

__host__
void Neuron::addIncomingInhibitorySynapse(Synapse &synapse) {
    incoming_inhibitory_synapses->append(synapse);
}
