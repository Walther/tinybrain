const { Neuron } = require('./neuron');

let weights = new Array(10).fill(1);
let bias = 0;

const n1 = new Neuron(weights, bias);
console.log(n1);
n1.randomize();
console.log(n1);
