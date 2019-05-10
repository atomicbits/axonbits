//
// Created by Peter Rigole on 2019-05-03.
//

#ifndef AXONBITS_NEURANETTEST_H
#define AXONBITS_NEURANETTEST_H

#include <assert.h>
#include "../Test.cu"
#include "../../NeuralNet.cuh"
#include "TestInputProcessor.cu"
#include "TestOutputProcessor.cu"

class NeuralNetTest : public Managed, public Test {

public:

    NeuralNetTest() : Test("NeuralNetTest") {}

    ~NeuralNetTest() {}

//    __host__
//    void hostTest() {
//
//        unsigned int nbOfNeurons = 1000000;
//        int nbOfSynapses = 1000;
//
//        NeuralNet* neuralNet = new NeuralNet(1);
//
//        Neuron** data_init;
//        cudaMallocManaged(&data_init, nbOfNeurons * sizeof(Neuron));
//
//        neuralNet->init();
//
//    }

    __host__
    void test() {

        printf("creating the neural net");

        unsigned int nbOfNeurons = 10000000;
        int nbOfSynapses = 100;

        NeuralNet* neuralNet = new NeuralNet(nbOfNeurons);

        NeuronProperties *properties = new NeuronProperties();

        Neuron neuronZero = Neuron(properties, 100, 0);
        neuralNet->addNeuron(neuronZero, 0);
        float activities[1] = { 0.9 };
        neuralNet->updateActivity(activities, 0, 0);

        /**
         * The results of this test in 32GB RAM:
         *
         * 1/ without synapses => process crashes between 4.100.000 and 4.200.000 neurons
         * 2/ with 100 synapses per neuron => process crashes between 100.000 and 200.000 neurons !!!
         */

        for (int i = 1; i < nbOfNeurons; ++i) {
            if (i % 100000 == 0) printf(" adding neuron %i", i);
            Neuron neuron = Neuron(properties, nbOfSynapses, 0);
            for (int j = 0; j < nbOfSynapses; ++j) {
                Synapse synapse = Synapse(0.5f, 0);
                neuron.addIncomingExcitatorySynapse(synapse);
            }
            neuralNet->addNeuron(neuron, i);
        }

        TestInputProcessor* inputProcessor = new TestInputProcessor(neuralNet);
        TestOutputProcessor* outputProcessor = new TestOutputProcessor(neuralNet);
        neuralNet->register10HzInputProcessor(inputProcessor);
        neuralNet->register10HzOutputProcessor(outputProcessor);

        neuralNet->init();

        for (int i = 0; i < 100; ++i) {
            printf("trial %i", i);
            neuralNet->trial();
        }

        delete (inputProcessor);
        delete (outputProcessor);

    }

};


#endif //AXONBITS_NEURANETTEST_H
