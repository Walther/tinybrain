const assert = require('assert');
const should = require('chai').should();
const { Sigmoid } = require('../src/sigmoid');
const sigmoid = new Sigmoid();

describe('Sigmoid', () => {
    describe('Forward', () => {
        it('Should be a function', () => {
            sigmoid.forward.should.be.a('Function');
        });
        it('Should return 0.7310585786300049 for 1', () => {
            sigmoid.forward(1).should.equal(0.7310585786300049);
        });
        it('Should return 0.5 for 0', () => {
            sigmoid.forward(0).should.equal(0.5);
        });
        it('Should return 0.2689414213699951 for -1', () => {
            sigmoid.forward(-1).should.equal(0.2689414213699951);
        });
    });
    describe('Backward', () => {
        it('Should be a function', () => {
            sigmoid.backward.should.be.a('Function');
        });
        it('Should return 0.19661193324148185 for 1', () => {
            sigmoid.backward(1).should.equal(0.19661193324148185);
        });
        it('Should return 0.25 for 0', () => {
            sigmoid.backward(0).should.equal(0.25);
        });
        it('Should return 0.19661193324148185 for -1', () => {
            sigmoid.backward(-1).should.equal(0.19661193324148185);
        });
    });
});
