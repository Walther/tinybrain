const assert = require('assert');
const should = require('chai').should();
const { Network } = require('../src/network');
const { Neuron } = require('../src/neuron');

class DummyNonLinearity {
    // Dummy linear activation function
    forward(value) {
        return value;
    }
    backward(value) {
        return 1;
    }
}

describe('Network', () => {
    let inputFeatures = 2;
    let hiddenLayers = 2;
    let outputClasses = 2;
    let initialWeights = new Array(inputFeatures).fill(1);
    let initialBias = 0;
    let nonLinearity = new DummyNonLinearity();
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
        it('DEV DUMMY: Should run a training round', () => {
            let testInput = [0, 1];
            network.randomize();
            network.doTrainingRound(testInput);
        });
    });
});
