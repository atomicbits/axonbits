# axonbits
An experimental asynchronous neural net implementation native on a CUDA GPU, based on the Leabra algorithms.

The idea of this implementation is that there is no matrix-based mathematical model between the neural net model and the GPU execution. Instead, we schedule the neural net execution directly on the GPU threads on the level of the neurons. This way each GPU core will process a number of neurons in each cycle. 

The advantage of this approach is that we skip the typical matrix math layer between the neural net model and the GPU execution with a one-on-one mapping. This way we are more flexible in playing around with our neural net model, as long as all execution can be expressed on the level of a single neuron. The disadvantage is that we cannot provide mathematical shortcuts across multiple neurons, as is done for example in the Leabra framework to implement cortical-cortical inhibition by using a winner-takes-all algorithm. We will have to implement this with normal neural inhibition behaviour, with a typical 15% neural activity overhead (as 15% of the cortical neurons are inhibitory on average).
