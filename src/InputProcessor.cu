//
// Created by Peter Rigole on 2019-05-03.
//

#include "InputProcessor.cuh"

InputProcessor::InputProcessor() {}

InputProcessor::InputProcessor(NeuralNet* neuralNet_init) : neuralNet(neuralNet_init) {}

void InputProcessor::setNeuralNet(NeuralNet* neuralNet_update) {
    neuralNet = neuralNet_update;
}

void InputProcessor::processInput() {

}
