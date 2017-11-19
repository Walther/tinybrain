class Sigmoid {
    forward(value) {
        return 1 / (1 + Math.exp(-value));
    }

    backward(value) {
        return this.forward(value) * (1 - this.forward(value));
    }
}

exports.Sigmoid = Sigmoid;
