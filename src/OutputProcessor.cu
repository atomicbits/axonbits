//
// Created by Peter Rigole on 2019-05-03.
//

#include "OutputProcessor.cuh"

OutputProcessor::OutputProcessor() {}

OutputProcessor::OutputProcessor(NeuralNet* neuralNet_init) : neuralNet(neuralNet_init) {}

void OutputProcessor::setNeuralNet(NeuralNet* neuralNet_update) {
    neuralNet = neuralNet_update;
}

void OutputProcessor::processOutput() {

}
