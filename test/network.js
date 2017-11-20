const assert = require('assert');
const should = require('chai').should();
const _ = require('lodash');
const { Network } = require('../src/network');
const { Neuron } = require('../src/neuron');
const { ReLU } = require('../src/relu');
const { Sigmoid } = require('../src/sigmoid');

describe('Network', () => {
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
    describe('Properties', () => {
        it('Should be an object', () => {
            network.should.be.an.instanceof(Object);
        });
        it('Should have hidden layers', () => {
            network.hiddenLayers.should.be.an.instanceof(Array);
        });
        it('Should have an output layer', () => {
            network.outputLayer.should.be.an.instanceof(Array);
        });
    });
    describe('Output layer', () => {
        it('Should have neurons', () => {
            network.outputLayer.map(neuron =>
                neuron.should.be.an.instanceof(Neuron)
            );
        });
    });
    describe('Hidden layers', () => {
        it('Should exist', () => {
            network.hiddenLayers.should.be.an.instanceof(Array);
        });
        it('Should have neurons', () => {
            network.hiddenLayers.map(layer =>
                layer.map(neuron => neuron.should.be.an.instanceof(Neuron))
            );
        });
    });
    describe('Randomize', () => {
        it('Should be a function', () => {
            network.randomize.should.be.an.instanceof(Function);
        });
        it('Should randomize the weights', () => {
            let initialString = JSON.stringify(network);
            network.randomize();
            let afterString = JSON.stringify(network);
            initialString.should.not.equal(afterString);
        });
    });
    describe('Forward pass', () => {
        it('Should be a function', () => {
            network.forwardPass.should.be.an.instanceof(Function);
        });
        it('Should calculate the final activations', () => {
            let testInput = [0, 1];
            network.randomize();
            let output = network.forwardPass(testInput);
            output.should.be.an.instanceof(Array);
        });
    });
    describe('Training round', () => {
        it('DEV DUMMY: Should run a training round', done => {
            // Assume output classification order [0, 1]
            let xorTarget = input => {
                if (input[0] === input[1]) {
                    return [1, 0]; // classification 0
                } else {
                    return [0, 1]; // classification 1
                }
            };
            let inputs = [[0, 0], [1, 1], [0, 1], [1, 0]];
            console.log(inputs.map(xorTarget));
            //let inputs = [[0, 1], [1, 0]];

            network.randomize();
            for (let round = 0; round < 1e6; round++) {
                let testInput = _.sample(inputs);
                let testTarget = xorTarget(testInput);
                network.doTrainingRound(testInput, xorTarget(testInput));
                //console.log(network.outputLayer[0]);
            }
            //console.log('---');
            //console.log(JSON.stringify(network.outputLayer, null, 2));
            // tests
            console.log(network.forwardPass([0, 0])); // 0 aka [1, 0]
            console.log(network.forwardPass([1, 1])); // 0 aka [1, 0]
            console.log(network.forwardPass([0, 1])); // 1 aka [0, 1]
            console.log(network.forwardPass([1, 0])); // 1 aka [0, 1]
            done();
        }).timeout(2 * 60 * 1000);
    });
});
