# Specification

TinyBrain is a naive, minimalistic, neural network implementation from scratch. It plays the board game Go.

## Motivation

I want to learn more about neural networks and how they work, so implementing one is a nice hands-on approach.

I also enjoy the board game Go. Neural networks have recently proven to be a great approach for this complex game. [^1] [^2]

Additionally, doing this project within the guidelines and specs, I can get credits for the *Algorithms and Data structures: Project* course.

## Implementation details

- Board size: 9x9
- Feature layers: 9
    - Current player's stones
    - Opponent's stones
    - 3 previous stone states for both players
    - Color of current player (for komi rule)
- Layer depth: 9
- Activation function: Rectified Linear Unit (ReLU)
- Initial weights: uniform random
- Learning method: reinforcement only

## Data structures used

- Artifical neuron: a structure with a weight matrix for inputs, bias, etc. 
- Arrays: 1D, 2D

## Algorithms used

- Neural network forward pass
- Neural network backpropagation

## Input and output

Practically, input for the progam is a representation of current game state and history. Initial plan is to use the SGF standard format.

Output is a single recommended move, as a coordinate.

Within the implementation, the neural network will get as an input an object with the following info:

- Board sized binary matrix of current player's stones
- Board sized binary matrix of opponent's stones
- 2*N board sized binary matrices of previous game states (one per player as above, per history level)
- Board sized binary matrix of all 1 if black is to play, all 0 if white is to play, due to komi rule

These are called feature layers.

Essentially, this results in the neural network's input object having (board size x feature layers) values.

## Time and space complexity analysis

Analyzing the O-notation of the neural network implementation might be a bit moot: input size will be constant and the processing within neural network should always involve the same amount of computations within the neurons and propagations, hence being O(1).

A more interesting insight to the computational weight of the network is the architecture: amount of computations required for processing depends on the size of individual layer, and amount of layers.

Original research on AlphaGo Zero [^2] used board size 19x19 and 17 feature layers, i.e. input size of 19x19x17. The original research used 20 (or 40) residual blocks, resulting in 39 (or 79) parametrized layers, plus two layers for the policy head and value head. This results in roughly (19x19x17)^39 (or ^79) raw multiplication computations for the neuron activations! Luckily with matrix multiplication operations instead of individual numbers, the actual operation count is closer to (19x19x17)x39 (or x79). (This also serves as a nice introduction to why the rise of powerful GPUs were so crucial for the development of neural networks)

Nevertheless, these numbers are way too large for the scope of running on student-budget consumer hardware compared to dedicated research hardware, not to forget the inefficiencies caused by hand-implementing everything from scratch, naively, without any top-of-the-line research optimizations present in existing neural network libraries.

Hence the scope of this project will be limited to something much smaller. See *Implementation details* for up-to-date numbers - the general formula of amount of computations is (board size x feature layers x depth).

## Original assigment questions

> Mitä algoritmeja ja tietorakenteita toteutat työssäsi

> Mitä ongelmaa ratkaiset ja miksi valitsit kyseiset algoritmit/tietorakenteet

> Mitä syötteitä ohjelma saa ja miten näitä käytetään

> Tavoitteena olevat aika- ja tilavaativuudet (m.m. O-analyysit)

[^3]

## References

[^1]: Mastering the game of Go with deep neural networks and tree search. Nature, 2016. [AlphaGo 2016](http://nature.com/articles/doi:10.1038/nature16961)

[^2]: Mastering the game of Go without human knowledge. Nature, 2016. [AlphaGo 2017](http://nature.com/articles/doi:10.1038/nature24270)

[^3]: Tiralabra 2017: [Dokumentaatio](https://github.com/TiraLabra/2017-syksy-periodi-2/wiki/Dokumentaatio)