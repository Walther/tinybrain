const assert = require('assert');
const should = require('chai').should();
const { ReLU } = require('../src/relu');

describe('ReLU', () => {
    let relu = new ReLU();
    describe('Properties', () => {
        it('Should have a forward function', () => {
            relu.forward.should.be.an('Function');
        });
        it('Should have a backward function', () => {
            relu.backward.should.be.an('Function');
        });
    });
    describe('Forward', () => {
        it('Should return value for value > 0', () => {
            relu.forward(1).should.equal(1);
        });
        it('Should return 0 for value < 0', () => {
            relu.forward(-1).should.equal(0);
        });
        it('Should return 0 for value == 0', () => {
            relu.forward(0).should.equal(0);
        });
    });
    describe('Backward', () => {
        it('Should return 1 for value >= 0', () => {
            relu.backward(1).should.equal(1);
        });
        it('Should return 0 for value < 0', () => {
            relu.backward(-1).should.equal(0);
        });
        it('Should return 0 for value == 0', () => {
            relu.backward(0).should.equal(0);
        });
    });
});
