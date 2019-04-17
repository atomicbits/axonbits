//
// Created by Peter Rigole on 2019-03-13.
//

#include "Neuron.cuh"

Neuron::Neuron() : id(0),
                   activity(0),
                   previous_activity(0),
                   long_term_avg_activity(0) {}

// Copy constructor
Neuron::Neuron(const Neuron &neuron) : id(neuron.id),
                                       activity(neuron.activity),
                                       previous_activity(neuron.previous_activity),
                                       long_term_avg_activity(neuron.long_term_avg_activity),
                                       properties(neuron.properties) {}

Neuron::~Neuron() {
    // cudaFree(data); // free the pointers!
}

// Get the id
__host__ __device__
unsigned long int Neuron::getId() const { return id; }
