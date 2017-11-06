const { softmax } = require('./softmax');

class Neuron {
    constructor(weights, bias) {
        this.weights = weights;
        this.bias = bias;
    }

    randomize() {
        let array = Array.from(
            { length: this.weights.length },
            () => Math.random() - 0.5
        );
        this.weights = array;
    }
}

exports.Neuron = Neuron;
