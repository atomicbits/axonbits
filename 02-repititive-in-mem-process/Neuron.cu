//
// Created by Peter Rigole on 2019-03-13.
//

#include "Neuron.cuh"

Neuron::Neuron() : id(0), v(0), group_firing_q_index(0) {}

// Copy constructor
Neuron::Neuron(const Neuron &s) : id(0), v(0), group_firing_q_index(0) {
}

Neuron::~Neuron() {
    // cudaFree(data); // free the pointers!
}

// Get the id
__host__ __device__
unsigned long int Neuron::getId() const { return id; }
