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
#include "../../util/Timer.cuh"

class NeuralNetTest : public Managed, public Test {

public:

    NeuralNetTest() : Test("NeuralNetTest") {}

    ~NeuralNetTest() {}

    __host__
    void test() {

        printf("creating the neural net\n");

        /**
         * About the number of neurons
         *
         * We get about 2.4 million neurons in 32GB RAM if they have 1000 incoming synapses each. The theoretical number
         * is a lot higher (4.25 million) and we don't know why the difference is so big yet. There must be some
         * overhead, but probably not so much.
         *
         * With 2.4 million neurons, we can:
         * - create 36 layers of size 256x256 (one neuron at each location)
         * - create 9 layers of size 192x192 where each location represents a cortical column (cc) with 6 excitatory and 1 inhibitory neuron
         * - create 20 layers of size 128x128 where each location is a cc with 6 exc. and 1 inh. neuron
         * with each neuron having 1000 incoming synapses on average in all tree situations.
         *
         */
        unsigned int nbOfNeurons = 100; // 2400000; // about 36 256x256 layers, or using 6 exc + 1 inh cort. columns: 9 192x192 cortical column layers (or 20 128x128 cc layers)
        int nbOfSynapses = 1000;

        NeuralNet* neuralNet = new NeuralNet(nbOfNeurons);
        printf("Before first init\n");
        neuralNet->init();
        printf("After first init\n");

        NeuronProperties *properties = new NeuronProperties();

        Neuron neuronZero = Neuron(properties, 100, 0);
        neuralNet->addNeuron(neuronZero, 0);
        float activities[1] = { 0.9 };
        neuralNet->updateActivity(activities, 0, 0);

        printf("\nEmpty neural net created\n");


        /**
         * The results of this test in 32GB RAM
         *
         * On the version using pointers in all arrays:
         *
         * 1/ without synapses => process crashes between 4.100.000 and 4.200.000 neurons
         * 2/ with only 100 synapses per neuron => process crashes between 100.000 and 200.000 neurons !!!
         *
         *
         * The version using no pointers in the arrays, but instead the raw objects, and the synapses use an
         * unsigned int (4 bytes) instead of a pointer reference (8 bytes) to the source neuron (the unsigned int
         * represents an offset in the global neurons array).
         *
         * 1/ with 1000 (!) synapses per neuron => process crashes between 2.400.000 and 2.500.000 neurons.
         * => the improvement is gigantic, just by dumping the pointers
         * It is, however, still a lot lower than the theoretical 4.25 million neurons that should fit in 32GB...
         *
         * Remark that these measurements are done with a bare application that loads the neurons and synapses into
         * memory, so without any surroundings infrastructure to interact with a client application that writes and
         * reads data into and out of the neural net! So, our limits may go down a bit in the future application.
         *
         */

        for (int i = 1; i < nbOfNeurons; ++i) {
            if (i % 100000 == 0) printf(" adding neuron %i\n", i);
            Neuron *neuron = new Neuron(properties, nbOfSynapses, 0); // The 'new' triggers the 'new Array' for the synapses inside the neuron object...
            // float randomActivity = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            // neuron->setExternalActivity(randomActivity);
            neuron->setExternalActivity(0.001f);
            for (int j = 0; j < nbOfSynapses; ++j) {
                // float randomWeight = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                Synapse synapse = Synapse(0.6f, 0);
                neuron->addIncomingExcitatorySynapse(synapse);
            }
            neuralNet->addNeuron(*neuron, i); // ToDo: add a factory method directly into the NeuralNet class to make creating the neuralnet more efficient.
            delete (neuron); // Doesn't delete the synapse arrays, which is good because they are referenced by the copied neurons in the global neuron array!

            neuralNet->init();
        }

        TestInputProcessor* inputProcessor = new TestInputProcessor(neuralNet);
        TestOutputProcessor* outputProcessor = new TestOutputProcessor(neuralNet);
        neuralNet->register10HzInputProcessor(inputProcessor);
        neuralNet->register10HzOutputProcessor(outputProcessor);

        neuralNet->init();

        Timer *timer = new Timer();
        double time;
        double sum = 0.0;
        int nbOfTrials = 100;
        for (int i = 0; i < nbOfTrials; ++i) {
            timer->reset();
            neuralNet->trial();
            time = timer->elapsed();
            sum += time;
            printf("trial %i (elapsed time: %f s)\n", i, time);
        }
        printf("average trial time is %f s\n", sum / nbOfTrials);

        checkCudaErrors();

        printf("\n");
        for (int i = 0; i < nbOfNeurons; ++i) {
            Neuron *neuron = neuralNet->getNeuron(i);
            float activity = neuron->getActivity(OddCycle);
            printf("%f, ", activity);
        }
        printf("\n\n");
        Neuron *neuron = neuralNet->getNeuron(nbOfNeurons / 2);
        for (Array<Synapse>::iterator j = neuron->getIncomingExcitatorySynapses()->begin(); j != neuron->getIncomingExcitatorySynapses()->end(); ++j) {
            Synapse synapse = *j;
            printf("%f, ", synapse.getWeight());
        }



        /**
         * Result on one Geforce GTX 1080 Ti and 32GB host RAM:
         *
         * average trial time is 0.499408 s
         *
         * About 5x too slow to be realtime (each trial represents 100ms).
         *
         */

        delete (timer);
        delete (neuralNet);
        delete (properties);
        delete (inputProcessor);
        delete (outputProcessor);

    }

};


#endif //AXONBITS_NEURANETTEST_H
