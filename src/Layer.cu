//
// Created by Peter Rigole on 2019-04-24.
//

#include "Layer.cuh"

Layer::Layer(unsigned int id_init, unsigned int xMaxIndex_init, unsigned int yMaxIndex_init) :
        id(id_init),
        xMaxIndex(xMaxIndex_init),
        yMaxIndex(yMaxIndex_init) {
    unsigned int nb_of_neurons = (xMaxIndex_init + 1) * (yMaxIndex_init + 1);
    neurons = new Array<Neuron>(nb_of_neurons);
}

Layer::~Layer() {
    // Destroying a layer destroys all its neurons!
    delete neurons;
}

// Get the id
__host__ __device__
unsigned long int Layer::getId() const {
    return id;
}

__host__ __device__
void Layer::setNeuron(unsigned int x, unsigned int y, Neuron* neuron) {
    neurons->set(neuron, xyToIndex(x, y));
}

__host__ __device__
Neuron* Layer::getNeuron(unsigned int x, unsigned int y) {
    return (*neurons)[xyToIndex(x, y)];
}

__host__ __device__
unsigned int Layer::xyToIndex(unsigned int x, unsigned int y) {
    if(x > xMaxIndex) assert(0);
    if(y > yMaxIndex) assert(0);
    return x * (yMaxIndex + 1) + y;
}
