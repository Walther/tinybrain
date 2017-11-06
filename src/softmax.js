const softmax = array => {
    let sigma = array
        .map(z => Math.pow(Math.E, z))
        .reduce((sum, value) => sum + value);
    return array.map(z => Math.pow(Math.E, z) / sigma);
};

exports.softmax = softmax;
