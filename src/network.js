const { Neuron } = require('./neuron');
const _ = require('lodash');

class Network {
    /**
     * Constructs a neural network with Neurons
     * @param {number} inputFeatures Number of input features, i.e. input vector size
     * @param {number} hiddenLayers Number of hidden layers
     * @param {number} outputClasses Number of output classes
     * @param {function} nonlinearity The nonlinearity function to use in the Neurons
     */
    constructor(inputFeatures, hiddenLayers, outputClasses, nonLinearity) {
        let initialWeights = new Array(inputFeatures).fill(1);
        let initialBias = 0;
        this.outputLayer = new Array(outputClasses)
            .fill()
            .map(item => new Neuron(initialWeights, initialBias, nonLinearity));

        this.hiddenLayers = new Array(hiddenLayers)
            .fill()
            .map(layer =>
                new Array(inputFeatures)
                    .fill()
                    .map(
                        item =>
                            new Neuron(
                                initialWeights,
                                initialBias,
                                nonLinearity
                            )
                    )
            );
        // Seal all the arrays
        Object.seal(this.outputLayer);
        this.hiddenLayers.map(layer => Object.seal(layer));
        Object.seal(this.hiddenLayers);
    }
    /**
     * Randomizes all the weights in the network
     */
    randomize() {
        this.hiddenLayers.map(layer => layer.map(neuron => neuron.randomize()));
        this.outputLayer.map(neuron => neuron.randomize());
    }
    /**
     * Computes a full forward pass of the network
     * @param {Array.<number>} input Input vector, as a number array
     * @returns {Array.<number>} Final activations at the output layer
     */
    forwardPass(input) {
        let hiddenOutput = this.hiddenLayers.reduce(
            this.activateNeurons,
            input
        );
        let finalOutput = this.activateNeurons(hiddenOutput, this.outputLayer);
        return finalOutput;
    }

    doTrainingRound(input, target, learningRate) {
        let predictions = this.forwardPass(input); // returns array of final output layer activations

        // Calculate partial derivatives for the layer's neurons with respect to the error
        let partials = (layer, nextLayer) =>
            layer.map((neuron, index) => {
                let error;
                if (nextLayer) {
                    // On hidden layer:
                    // assume partial term saved on neuron state on previous backprop
                    error = nextLayer
                        .map(
                            neuron =>
                                neuron.weights[index] * neuron.getPartial()
                        )
                        .reduce((sum, value) => sum + value);
                } else {
                    // On output layer:
                    error = predictions[index] - target[index];
                }
                let partial =
                    error *
                    neuron.nonlinearity.backward(neuron.getActivation());
                neuron.setPartial(partial);
            });

        partials(this.outputLayer, null); // ugly side-effect code!

        for (let index = this.hiddenLayers.length - 1; index >= 0; index--) {
            if (index === this.hiddenLayers.length - 1) {
                // Last hidden layer, use output layer
                partials(this.hiddenLayers[index], this.outputLayer);
            } else {
                // Else, in general case, use the next layer
                partials(
                    this.hiddenLayers[index],
                    this.hiddenLayers[index + 1]
                );
            }
        }

        // We now have the partials, let's update the weights to fix errors
        // Start with output layer alone
        let updateWeights = layer =>
            layer.map(neuron => {
                neuron.setWeights(
                    neuron.weights.map((weight, index) => {
                        return (
                            weight +
                            -1 *
                                learningRate *
                                neuron.getInputs()[index] *
                                neuron.getPartial()
                        );
                    })
                );
                neuron.setBias(
                    neuron.bias + -1 * learningRate * neuron.getPartial()
                );
            });
        updateWeights(this.outputLayer);
        this.hiddenLayers.forEach(updateWeights);
        let totalError =
            1 /
            predictions.length *
            _.sum(
                predictions.map((value, index) => {
                    return Math.pow(value - target[index], 2);
                })
            );
        return totalError;
    }

    /**
     * Helper function for neuron activations, used in forwardPass() as the reducer
     * @param {Array.<number>} currentInput Input vector we're currently interested in
     * @param {Array.<Neuron>} layer Layer of neurons we're running the activations of
     */
    activateNeurons(currentInput, layer) {
        // For each layer, activate all neurons & return their values
        return layer.map(neuron => neuron.activate(currentInput));
    }
}

exports.Network = Network;
