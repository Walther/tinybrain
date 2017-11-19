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

    dummyTeacherOutput(input) {
        // Dummy teacher that wants the network to converge to always return first option
        // TODO: remove, this is only for dev purposes!
        let array = new Array(input.length).fill(0);
        array[0] = 1;
        return array;
    }

    doTrainingRound(input) {
        let predictions = this.forwardPass(input); // returns array of final output layer activations
        let targets = this.dummyTeacherOutput(input); // assume method of getting the desired output

        // Calculate partial derivatives for the layer's neurons with respect to the error
        let partials = (layer, nextLayer) =>
            layer.map((neuron, index) => {
                let partial;
                if (nextLayer) {
                    // On hidden layer:
                    // assume error term saved on neuron state on previous backprop
                    partial = nextLayer
                        .map(
                            neuron => neuron.weights[index] * neuron.getError()
                        )
                        .reduce((sum, value) => sum + value);
                } else {
                    // On output layer:
                    partial = targets[index] - predictions[index];
                }
                let error =
                    partial *
                    neuron.nonlinearity.backward(neuron.getActivation());
                neuron.setError(error);
            });

        partials(this.outputLayer, null); // ugly side-effect code!
        console.log('Output layer: ' + JSON.stringify(this.outputLayer));

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

        console.log('Hidden layers: ' + JSON.stringify(this.hiddenLayers));

        // What about bias?
        let biasDelta = 0;

        // And then generalize this for all neurons at that layer, and the previous layers
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
