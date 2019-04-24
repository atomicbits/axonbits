
#include <cuda_runtime.h>

#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#include "NeuralNet.cuh"

sig_atomic_t volatile g_running = 1;

void sig_handler(int signum)
{
    if (signum == SIGINT)
        g_running = 0;
}


__global__
void add_input_spikes(NeuralNet *elem) {
    return;
}

__global__
void push_spikes(NeuralNet *elem) {
    return;
}

__global__
void generate_spikes(NeuralNet *elem) {
    return;
}

void launch_add_input_spikes(NeuralNet *elem) {
    add_input_spikes<<< 1, 1 >>>(elem);
    cudaDeviceSynchronize();
}

void launch_push_spikes(NeuralNet *elem) {
    push_spikes<<< 1, 1 >>>(elem);
    cudaDeviceSynchronize();
}

void launch_generate_spikes(NeuralNet *elem) {
    generate_spikes<<< 1, 1 >>>(elem);
    cudaDeviceSynchronize();
}


int main(int argc, char **argv)
{

    NeuralNet *neuralNet = new NeuralNet(5);

    signal(SIGINT, &sig_handler);

    while (g_running) {
        launch_add_input_spikes(neuralNet);
        launch_push_spikes(neuralNet);
        launch_generate_spikes(neuralNet);
    }

    printf("exiting safely\n");

//    printf("On host (after by-pointer): name=%s, value=%d\n", e->name.c_str(), e->value);

    delete neuralNet;

    cudaDeviceReset();

    return 0;
}
