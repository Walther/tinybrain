# Specification

TinyBrain is a naive, minimalistic, neural network implementation from scratch.

## Motivation

I want to learn more about neural networks and how they work, so implementing
one is a nice hands-on approach.

Additionally, doing this project within the guidelines and specs, I can get
credits for the _Algorithms and Data structures: Project_ course.

## Data structures used

* Artifical neuron: a structure with a weight matrix for inputs, bias, etc.
* Arrays: 1D, 2D

## Algorithms used

* Neural network forward pass
* Neural network backpropagation
* Stochastic gradient descent towards minimum error

## Input and output

The inputs are fixed-size per given neural network architecture: e.g. the input
for a network approximating the XOR function will always take an input array of
length two. The outputs are fixed-size too, determined at the time of designing
a network.

## Time and space complexity analysis

Analyzing the O-notation of the neural network implementation might be a bit
moot: input size will be constant and the processing within neural network
should always involve the same amount of computations within the neurons and
propagations, hence being O(1).

A more interesting insight to the computational weight of the network is the
architecture: amount of computations required for processing depends on the size
of individual layer, and amount of layers.

An example architecture of input vector size two, two neurons per layer, two
hidden layers, output layer of two classifications, the network will have
`2*2*2*2 = 16` multiplications per processing of an input.

## Original assigment questions

> Mitä algoritmeja ja tietorakenteita toteutat työssäsi

> Mitä ongelmaa ratkaiset ja miksi valitsit kyseiset algoritmit/tietorakenteet

> Mitä syötteitä ohjelma saa ja miten näitä käytetään

> Tavoitteena olevat aika- ja tilavaativuudet (m.m. O-analyysit)

[Tiralabra 2017](https://github.com/TiraLabra/2017-syksy-periodi-2/wiki/Dokumentaatio)
