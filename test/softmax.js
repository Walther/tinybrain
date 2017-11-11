const assert = require('assert');
const should = require('chai').should();
const { softmax } = require('../src/softmax');

describe('Softmax', () => {
    it('Should be a function', () => {
        softmax.should.be.a('Function');
    });
    it('Should return an array', () => {
        softmax([1, 1]).should.be.an('Array');
    });
    it('Should return [0.5, 0.5] for input of [1,1]', () => {
        softmax([1, 1])[0].should.be.approximately(0.5, 1e-6) &&
            softmax([1, 1])[1].should.be.approximately(0.5, 1e-6);
    });
    it('Should return [1] for input of [0]', () => {
        softmax([0]).should.deep.equal([1]);
    });
    it('Should return [1] for input of [-1]', () => {
        softmax([-1]).should.deep.equal([1]);
    });
    it('Should return [0.2689414213699951, 0.7310585786300048] for input of [1,2]', () => {
        softmax([1, 2])[0].should.equal(0.2689414213699951) &&
            softmax([1, 2])[1].should.equal(0.7310585786300048);
    });
    it('Should return [0.11920292202211756, 0.8807970779778824] for input of [-1,1]', () => {
        softmax([-1, 1])[0].should.equal(0.11920292202211756) &&
            softmax([-1, 1])[1].should.equal(0.8807970779778824);
    });
});
