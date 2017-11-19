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
        /* Sadly, we need to store some dirty state:
         * When doing a forward pass, we store the sum(weights * inputs)
         * and the input array this neuron got during that pass.
         * This is required for the backpropagation stage, for computing deltas.
         * See: [Delta Rule](https://enwp.org/Delta_rule)
         */
        this.activation = 0;
        this.inputs = new Array(weights.length).fill(0);
        this.partial = 0;
        // Closest we get to a real array in JS
        Object.seal(this.weights);
        Object.seal(this.inputs);
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
     * weights and bias. Stores the sum(weights * inputs) and input to the
     * neuron state, for use in the backpropagation pass.
     * @param {[number]} input Input vector for the neuron
     * @returns {number} Activation of the neuron
     */
    activate(input) {
        this.inputs = input;
        let weightedSum = input
            .map((value, index) => value * this.weights[index])
            .reduce((value, sum) => sum + value);
        this.activation = this.nonlinearity.forward(weightedSum + this.bias);
        return this.activation;
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

    setPartial(partial) {
        this.partial = partial;
    }

    /**
     * Getter for the Neuron's weighted input sum. Used for backpropagation
     * pass
     * @returns {number} sum(weights * inputs)
     */
    getActivation() {
        return this.Activation;
    }

    /** Getter for the Neuron's inputs at the current forward pass.
     * Used for backpropagation pass.
     * @returns {[number]} inputs´
     */
    getInputs() {
        return this.inputs;
    }

    getPartial() {
        return this.partial;
    }
}

exports.Neuron = Neuron;
