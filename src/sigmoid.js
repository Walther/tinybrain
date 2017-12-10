/**
 * Sigmoid function. Squashes the input (any real number) to the range -1..1. Used as a nonlinear transfer function in neurons. See https://en.wikipedia.org/wiki/Sigmoid_function
 */
class Sigmoid {
    /**
     * Forward pass of a Sigmoid transfer with given value
     * @param {number} value Number to pass through the sigmoid function
     */
    forward(value) {
        return 1 / (1 + Math.exp(-value));
    }

    /**
     * Derivative of the sigmoid function. Used in the backpropagation phase of learning.
     * @param {number} value Number to pass through the derivative of sigmoid function
     */
    backward(value) {
        return this.forward(value) * (1 - this.forward(value));
    }
}

exports.Sigmoid = Sigmoid;
