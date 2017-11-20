const assert = require('assert');
const should = require('chai').should();
const _ = require('lodash');
const { Network } = require('../src/network');
const { Neuron } = require('../src/neuron');
const { ReLU } = require('../src/relu');
const { Sigmoid } = require('../src/sigmoid');

describe('XOR: output as classification', () => {
    let inputFeatures = 2;
    let hiddenLayers = 1;
    let outputClasses = 2;
    let initialWeights = new Array(inputFeatures).fill(1);
    let initialBias = 0;
    let nonLinearity = new Sigmoid();
    let network = new Network(
        inputFeatures,
        hiddenLayers,
        outputClasses,
        nonLinearity
    );

    describe('Training round', () => {
        it('Should learn the XOR problem', done => {
            //TODO: Somehow this seems to work when using Sigmoid, but not ReLU
            // What is wrong with the learning algo?

            // Assume output classification order [0, 1]
            let xorTarget = input => {
                if (input[0] === input[1]) {
                    return [1, 0]; // classified as 0
                } else {
                    return [0, 1]; // classified as 1
                }
            };
            let inputs = [[0, 0], [1, 1], [0, 1], [1, 0]];

            network.randomize();
            let rounds = 1e6;
            for (let round = 0; round < rounds; round++) {
                let testInput = _.sample(inputs);
                let testTarget = xorTarget(testInput);
                let learningRate = 0.1;
                let totalError = network.doTrainingRound(
                    testInput,
                    xorTarget(testInput),
                    learningRate
                );
                if (round % (rounds / 10) === 0) {
                    console.log('Total error: ' + totalError);
                }
            }

            let testOutputs = [
                network.forwardPass([0, 0]),
                network.forwardPass([1, 1]),
                network.forwardPass([0, 1]),
                network.forwardPass([1, 0])
            ];

            console.log('Expecting: ' + JSON.stringify(inputs.map(xorTarget)));
            console.log(
                'Got: ' +
                    JSON.stringify(
                        testOutputs.map(valuepair => valuepair.map(Math.round))
                    )
            );
            done();
        }).timeout(1 * 60 * 1000);
    });
});
