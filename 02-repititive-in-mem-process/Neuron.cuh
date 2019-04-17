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

    // Get the id
    __host__ __device__
    unsigned long int getId() const;

private:
    unsigned long int id;
    float activity;
    float previous_activity;
    float long_term_avg_activity;
    Synapse *incoming_synapses[];
    NeuronProperties *properties;
};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
