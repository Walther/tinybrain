/**
 * Softmax function. Normalizes the given input array so that all numbers are between 0..1 and their sum is 1. See https://en.wikipedia.org/wiki/Softmax_function
 * @param {Array.<number>} array Array of numbers to run through the softmax function
 */
const softmax = array => {
    let sigma = array
        .map(z => Math.pow(Math.E, z))
        .reduce((sum, value) => sum + value);
    return array.map(z => Math.pow(Math.E, z) / sigma);
};

exports.softmax = softmax;
