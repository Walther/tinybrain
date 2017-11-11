const { softmax } = require('./softmax');

class Neuron {
    constructor(weights, bias, nonlinearity) {
        this.weights = weights;
        this.bias = bias;
        this.nonlinearity = nonlinearity;
    }

    randomize() {
        let array = Array.from(
            { length: this.weights.length },
            () => Math.random() - 0.5
        );
        this.weights = array;
    }

    activate(input) {
        return this.nonlinearity.forward(
            input
                .map((value, index) => value * this.weights[index])
                .reduce((value, sum) => sum + value)
        );
    }
}

exports.Neuron = Neuron;
