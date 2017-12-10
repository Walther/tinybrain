const assert = require('assert');
const should = require('chai').should();
const { Neuron } = require('../src/neuron');

class DummyActivationFunction {
    // Dummy linear activation function
    forward(value) {
        return value;
    }
    backward(value) {
        return 1;
    }
}

describe('Neuron', () => {
    let features = 10;
    let weights = new Array(features).fill(1);
    let bias = 0;
    let activationFunction = new DummyActivationFunction();
    let n1 = new Neuron(weights, bias, activationFunction);

    before(() => {
        n1 = new Neuron(weights, bias, activationFunction);
    });

    describe('Properties', () => {
        it('Should be an object', () => n1.should.be.an.instanceof(Neuron));
        it('Should have a weights array', () =>
            n1.weights.should.be.an('Array'));
        it('Should have a nonlinearity function object', () =>
            n1.nonlinearity.should.be.a('Object'));

        it('Should have an activation function', () =>
            n1.activate.should.be.an.instanceof(Function));
        it('Should have a randomization function', () =>
            n1.randomize.should.be.an.instanceof(Function));
    });

    describe('Randomize', () => {
        it('Should randomize the weights', () => {
            let originalWeights = n1.weights;
            n1.randomize();
            n1.weights.should.not.equal(originalWeights);
        });
    });

    describe('Nonlinearity', () => {
        it('Should have a nonlinearity function, forward pass', () =>
            n1.nonlinearity.forward.should.be.an.instanceof(Function));
        it('Should have a nonlinearity function, backward pass', () =>
            n1.nonlinearity.backward.should.be.an.instanceof(Function));
    });

    describe('Activate', () => {
        let testInputAllOnes = new Array(features).fill(1);
        let result = n1.activate(testInputAllOnes);
        it('Should calculate the activation', () => {
            result.should.be.a('Number');
        });
        it('Should calculate the activation correctly: all ones', () => {
            result.should.equal(10);
        });

        let testInputAllZeroes = new Array(features).fill(0);
        let result2 = n1.activate(testInputAllZeroes);
        it('Should calculate the activation correctly: all zeroes', () => {
            result2.should.equal(0);
        });
    });
    describe('Setters', () => {
        it('Should have a weights setter', () => {
            let newWeights = [...Array(features).keys()];
            n1.setWeights(newWeights);
            n1.weights.should.deep.equal(newWeights);
        });
        it('Should have a bias setter', () => {
            let newBias = 5;
            n1.setBias(newBias);
            n1.bias.should.equal(newBias);
        });
        it('Should have a partial setter', () => {
            let newPartial = 5;
            n1.setPartial(newPartial);
            n1.partial.should.equal(newPartial);
        });
    });
    describe('Getters', () => {
        it('Should have a inputs getter', () => {
            let inputs = [...Array(features).keys()];
            n1.activate(inputs);
            n1.inputs.should.deep.equal(inputs);
        });
        it('Should have a partial getter', () => {
            let newPartial = 5;
            n1.setPartial(newPartial);
            n1.getPartial().should.equal(newPartial);
        });
        it('Should have an activation getter', () => {
            let inputs = [...Array(features).keys()];
            n1.activate(inputs);
            n1.getActivation().should.equal(290);
        });
    });
});
