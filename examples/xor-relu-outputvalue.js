const assert = require('assert');
const should = require('chai').should();
const _ = require('lodash');
const { Network } = require('../src/network');
const { Neuron } = require('../src/neuron');
const { ReLU } = require('../src/relu');
const { Sigmoid } = require('../src/sigmoid');

describe('XOR: ReLU, output as value', () => {
    let inputFeatures = 2;
    let hiddenLayers = 1;
    let outputClasses = 1;
    let initialWeights = new Array(inputFeatures).fill(1);
    let initialBias = 0;
    let nonLinearity = new ReLU();
    let network = new Network(
        inputFeatures,
        hiddenLayers,
        outputClasses,
        nonLinearity
    );

    describe('Training round', () => {
        it('Should learn the XOR problem', done => {
            // Assume output as single value
            let xorTarget = input => {
                if (input[0] === input[1]) {
                    return [0];
                } else {
                    return [1];
                }
            };
            let inputs = [[0, 0], [1, 1], [0, 1], [1, 0]];

            network.randomize();
            let rounds = 1e6;
            for (let round = 0; round < rounds; round++) {
                let testInput = _.sample(inputs);
                let testTarget = xorTarget(testInput);
                let learningRate = 0.01;
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

            let expected = inputs.map(xorTarget);
            console.log('Expected: ' + JSON.stringify(expected));
            let got = testOutputs.map(vector => vector.map(Math.round));
            console.log('Got: ' + JSON.stringify(got));
            expected.should.deep.equal(got);
            done();
        }).timeout(1 * 60 * 1000);
    });
});
