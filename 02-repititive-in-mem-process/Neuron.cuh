//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H

#include "Managed.cuh"
#include "NeuronProperties.cuh"
#include "Synapse.cuh"

class NeuronProperties; // forward declaration to cope with cyclic dependency
class Synapse; // forward declaration to cope with cyclic dependency

class Neuron : public Managed {
public:
    // Default Constructor
    Neuron();

    // Copy constructor
    Neuron(const Neuron&);

    // Destructor
    ~Neuron();

    Neuron(unsigned long int neuronId, NeuronProperties* neuronProperties, unsigned int max_nb_incoming_synapses);

    // Get the id
    __host__ __device__
    unsigned long int getId() const;

    __host__ __device__
    float getActivity() const;

    __host__ __device__
    float getPreviousActivity() const;

    __host__ __device__
    float getLongTimeAverageActivity() const;

//    __host__ __device__
//    Synapse** getIncomingSynapses();

    __host__ __device__
    NeuronProperties* getProperties() const;

private:
    unsigned long int id;
    float activity;
    float previous_activity;
    float long_time_avg_activity;
    NeuronProperties *properties;
    Synapse* incoming_synapses;
    unsigned int nb_synapses = 0;
    unsigned int max_nb_synapses;
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
