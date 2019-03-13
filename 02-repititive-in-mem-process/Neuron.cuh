//
// Created by Peter Rigole on 2019-03-13.
//

#ifndef INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
#define INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H


class Neuron {
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
    float v;
    unsigned short int group_firing_q_index;
    // Synapse *outgoing_synapses[];
    // NeuronFiringQueue *incoming_firing_q;
    // NeuronFiringQueue *incoming_firing_q_history;
    // updateFunction();

};


#endif //INC_02_REPITITIVE_IN_MEM_PROCESS_NEURON_H
