class ReLU {
    forward(value) {
        return Math.max(0, value);
    }

    backward(value) {
        // Derivative, for backpropagation
        if (value > 0) {
            return 1;
        } else return 0;
    }
}

exports.ReLU = ReLU;
