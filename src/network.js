const { Neuron } = require('./neuron');

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
     * @param {[number]} input Input vector, as a number array
     * @returns {[number]} Final activations at the output layer
     */
    forwardPass(input) {
        let hiddenOutput = this.hiddenLayers.reduce(
            this.activateNeurons,
            input
        );
        let finalOutput = this.activateNeurons(hiddenOutput, this.outputLayer);
        return finalOutput;
    }
    backPropagate() {
        // TODO
        throw new Error('NOT IMPLEMENTED YET');
    }

    /**
     * Helper function for neuron activations, used in forwardPass() as the reducer
     * @param {[number]} currentInput Input vector we're currently interested in
     * @param {[Neuron]} layer Layer of neurons we're running the activations of
     */
    activateNeurons(currentInput, layer) {
        // For each layer, activate all neurons & return their values
        return layer.map(neuron => neuron.activate(currentInput));
    }
}

exports.Network = Network;