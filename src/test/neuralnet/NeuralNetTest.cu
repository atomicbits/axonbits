//
// Created by Peter Rigole on 2019-05-03.
//

#ifndef AXONBITS_NEURANETTEST_H
#define AXONBITS_NEURANETTEST_H

#include <assert.h>
#include "../Test.cuh"
#include "../../NeuralNet.cuh"
#include "TestInputProcessor.cu"
#include "TestOutputProcessor.cu"

class NeuralNetTest : public Managed, public Test {

public:

    NeuralNetTest() : Test(TestClass::neuralnettest) {}

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
    void hostTest() {

        printf("creating the neural net");

        unsigned int nbOfNeurons = 10000000;
        int nbOfSynapses = 1;

        NeuralNet* neuralNet = new NeuralNet(nbOfNeurons);

        NeuronProperties *properties = new NeuronProperties();

        Neuron *neuronZero = new Neuron(0, properties, 100, 0);
        neuralNet->addNeuron(neuronZero);
        float activities[1] = { 0.9 };
        neuralNet->updateActivity(activities, 0, 0);

        for (int i = 1; i < nbOfNeurons; ++i) {
            if (i % 100000 == 0) printf(" adding neuron %i", i);
            Neuron *neuron = new Neuron(i, properties, nbOfSynapses, 0);
//            for (int j = 0; j < nbOfSynapses; ++j) {
//                Synapse* synapse = new Synapse(0.5, neuronZero);
//                neuron->addIncomingExcitatorySynapse(synapse);
//            }
            neuralNet->addNeuron(neuron);
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

    __device__
    void deviceTest() {}

    __host__
    const char* getName() {
        return "NeuralNetTest";
    }

};


#endif //AXONBITS_NEURANETTEST_H
