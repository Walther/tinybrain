const { softmax } = require('./softmax');

class Neuron {
    /**
     * An artificial neuron.
     * @param {[number]} weights Array of the initial weights
     * @param {number} bias Initial bias
     * @param {function} nonlinearity Nonlinearity function, e.g. ReLU
     */
    constructor(weights, bias, nonlinearity) {
        this.weights = weights;
        this.bias = bias;
        this.nonlinearity = nonlinearity;
        Object.seal(this.weights); // Closest we get to a real array in JS
        Object.seal(this);
    }

    /**
     * Randomizes the neuron's weights
     */
    randomize() {
        let array = Array.from(
            { length: this.weights.length },
            () => Math.random() - 0.5
        );
        this.weights = array;
    }

    /**
     * Calculates the neuron's activation for the given input based on current
     * weights and bias
     * @param {[number]} input Input vector for the neuron
     * @returns {number} Activation of the neuron
     */
    activate(input) {
        return this.nonlinearity.forward(
            input
                .map((value, index) => value * this.weights[index])
                .reduce((value, sum) => sum + value)
        );
    }

    /**
     * Setter for the Neuron weights array
     * @param {[number]} weights List of weights to use for the Neuron
     */
    setWeights(weights) {
        this.weights = weights;
    }

    /**
     * Setter for the Neuron bias value
     * @param {number} bias Bias to use for the Neuron
     */
    setBias(bias) {
        this.bias = bias;
    }
}

exports.Neuron = Neuron;
