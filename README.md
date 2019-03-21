# FeedForwardNeuralNetwork
This is a feed forward neural network programmed from scratch in python, designed to recognise unique handwritten digits (from 0-9).
This project was to understand how feed forward and other neural networks perform in a mathematical environment instead of the conventional libraries e.g. tensorflow, pytorch etc. 

# How it works?
The feed forward algorithm essentially feeds the weights and inputs from each layer to the next until it reaches towards the output layer (where it will begin the backpropagation algorithm which will be mentioned later).

Feed forwards Algorithm:
In this context of this code, the network archetecture is {784, 16, 16, 10}. Each number in each index corresponds to the number of nodes or 'neurons'. The first index is the number of input nodes, second and third are the hidden layers and the last layer is the output layer. In between each layer (e.g. inputs and hiddenlayer1, hiddenlayer1 and hiddenlayer2 etc) there are values that are called 'weights' these serve a purpose weight the 'inputs' of each layer; these weights can be thought as synapses in the brains of living creatures. Mathematically in the first hidden layer of nodes (neurons), gather the weights ([previous input values from each node, next number of nodes/neurons])



