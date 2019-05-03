//
// Created by Peter Rigole on 2019-04-24.
//

#ifndef AXONBITS_LAYER_H
#define AXONBITS_LAYER_H

#include "Managed.cuh"
#include "Neuron.cuh"

class Layer : Managed {
public:

    Layer(unsigned int id_init, unsigned int xMaxIndex_init, unsigned int yMaxIndex_init);

    ~Layer();

    // Get the id
    __host__ __device__
    unsigned long int getId() const;

    __host__ __device__
    void setNeuron(unsigned int x, unsigned int y, Neuron* neuron);

    __host__ __device__
    Neuron* getNeuron(unsigned int x, unsigned int y);

private:

    __host__ __device__
    unsigned int xyToIndex(unsigned int x, unsigned int y);

    const unsigned long int id;
    const unsigned int xMaxIndex; // max x index (NOT max x size!)
    const unsigned int yMaxIndex; // max y index (NOT max y size!)
    Array<Neuron>* neurons;

};


#endif //AXONBITS_LAYER_H
