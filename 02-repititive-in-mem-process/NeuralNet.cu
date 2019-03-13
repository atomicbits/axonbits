//
// Created by Peter Rigole on 2019-03-13.
//
#include "NeuralNet.cuh"

NeuralNet::NeuralNet() : length(0), data(0) {}

// Constructor for C-string initializer
NeuralNet::NeuralNet(const char *s) : length(0), data(0) {
}

// Copy constructor
NeuralNet::NeuralNet(const NeuralNet &s) : length(0), data(0) {
}

NeuralNet::~NeuralNet() { cudaFree(data); }

// Get the name
__host__ __device__
char *NeuralNet::name() const { return data; }
