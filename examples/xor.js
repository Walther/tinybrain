const _ = require('lodash');
const commandLineArgs = require('command-line-args');

const { Network } = require('../src/network');
const { Neuron } = require('../src/neuron');
const { ReLU } = require('../src/relu');
const { Sigmoid } = require('../src/sigmoid');

// Helper definitions for XOR
const xorTargetClassification = input => {
    // Assume output classification order [0, 1]
    if (input[0] === input[1]) {
        return [1, 0]; // classified as 0
    } else {
        return [0, 1]; // classified as 1
    }
};

const xorTargetValue = input => {
    if (input[0] === input[1]) {
        return [0];
    } else {
        return [1];
    }
};

const inputs = [[0, 0], [1, 1], [0, 1], [1, 0]];

// CLI option parsing

const optionDefinitions = [
    { name: 'layers', alias: 'l', type: Number },
    { name: 'relu', alias: 'r', type: Boolean },
    { name: 'sigmoid', alias: 's', type: Boolean },
    { name: 'classification', alias: 'c', type: Boolean },
    { name: 'value', alias: 'v', type: Boolean },
    { name: 'epochs', alias: 'e', type: Number },
    { name: 'rate', alias: 'a', type: Number },
    { name: 'plot', alias: 'p', type: Boolean }
];
const options = commandLineArgs(optionDefinitions);

const inputFeatures = 2;
const hiddenLayers = options.layers;
let outputClasses;
let xorTarget;
if (options.classification) {
    outputClasses = 2;
    xorTarget = xorTargetClassification;
} else if (options.value) {
    outputClasses = 1;
    xorTarget = xorTargetValue;
}
let nonLinearity;
if (options.relu) {
    nonLinearity = new ReLU();
} else if (options.sigmoid) {
    nonLinearity = new Sigmoid();
}
const epochs = Number(options.epochs);
const learningRate = options.rate;
const plot = options.plot;

let network = new Network(
    inputFeatures,
    hiddenLayers,
    outputClasses,
    nonLinearity
);

network.randomize();
let totalError;
for (let round = 0; round < epochs; round++) {
    let testInput = _.sample(inputs);
    let testTarget = xorTarget(testInput);
    totalError = network.doTrainingRound(
        testInput,
        xorTarget(testInput),
        learningRate
    );
    if (plot) {
        console.log(totalError);
    }
}

let testOutputs = [
    network.forwardPass([0, 0]),
    network.forwardPass([1, 1]),
    network.forwardPass([0, 1]),
    network.forwardPass([1, 0])
];

if (!plot) {
    console.log('Taught the network with ' + epochs + ' training rounds.');
    console.log('Total error at the end of training: ' + totalError);
    let expected = inputs.map(xorTarget);
    console.log('Expecting: ' + JSON.stringify(expected));
    let got = testOutputs.map(valuepair => valuepair.map(Math.round));
    console.log('Got: ' + JSON.stringify(got));
}
