/**
 * Rectified Linear Unit. Used as a nonlinear transfer function in neurons. See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
class ReLU {
    /**
     * Forward pass of a ReLU transfer with given value
     * @param {number} value Number to pass through the ReLU function
     */
    forward(value) {
        return Math.max(0, value);
    }

    /**
     * Derivative of the ReLU. Used in the backpropagation phase of learning.
     * @param {number} value Number to pass through the derivative of the ReLU function
     */
    backward(value) {
        if (value > 0) {
            return 1;
        } else return 0;
    }
}

exports.ReLU = ReLU;
