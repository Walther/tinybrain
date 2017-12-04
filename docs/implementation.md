# Implementation

## Neuron

A `Neuron` is a simple object representing an artificial neuron. Its constructor
requires the initial weights of the neuron, initial bias of the neuron, and the
nonlinearity aka transfer function used in it.

A `Neuron` has the following important methods:

* `randomize` randomizes the weights and the bias of the neuron. This is linear
  with respect to the size of the neuron, i.e. length of its `weights`.
* `activate` calculates the activation of the neuron based on its input,
  `weights` and the transfer function. This is linear with respect to the size
  of the neuron.

## Transfer functions

`ReLU`, `Sigmoid`

A transfer function, used to introduce a non-linearity to the neurons and hence
give the network the ability to learn.

Constant run time.

## Network

A `Network` is an object consisting of layers and neurons within them.

A `Network` has the following important methods:

* `randomize` is a helper calling `randomize` to each of its neurons. Causes
  `layers * neurons_per_layer * weights_per_neuron` computations of randomness
* `doTrainingRound` is complicated, complexity analysis TODO
