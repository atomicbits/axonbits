//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H

#include "Managed.cuh"
#include "NeuronProperties.cuh"
#include "Synapse.cuh"
#include "util/Array.cu"
#include "Phase.cu"
#include "CycleParity.cu"


class NeuronProperties; // forward declaration to cope with cyclic dependency
class Synapse; // forward declaration to cope with cyclic dependency

class Neuron : public Managed {
public:

    /**
     * Creates a memory managed neuron with the given id and neuron properties. It reserves the space to store
     * max_nb_incoming_synapses number of synapse pointers in a managed array.
     *
     * @param neuronId
     * @param neuronProperties
     * @param max_nb_incoming_synapses
     */
    Neuron(unsigned long int neuronId, NeuronProperties* neuronProperties, unsigned int max_nb_incoming_synapses);

    // Destructor
    ~Neuron();

    // Get the id
    __host__ __device__
    unsigned long int getId() const;

    __host__ __device__
    float getActivity() const;

    __host__ __device__
    void updateActivity(float activity_update);

    __host__ __device__
    float getPreviousActivity() const;

    __host__ __device__
    float getLongTimeAverageActivity() const;

    __host__ __device__
    Array<Synapse>* getIncomingSynapses() const;

    __host__ __device__
    const NeuronProperties* getProperties() const;

    __host__ __device__
    void addIncomingSynapse(Synapse* synapse);

private:
    unsigned long int id;
    float activity; // y(t)
    float previous_activity; // y(t-1)
    float long_time_avg_activity; // y_l
    const NeuronProperties* properties;
    Array<Synapse>* incoming_synapses;
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
