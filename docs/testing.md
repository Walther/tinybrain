# Testing

## Unit tests

To run the unit tests, run:

```
yarn run test
```

## Neural network training testing

Pre-requirements:

* `node` version 8 or above
* `yarn`
* `gnuplot`

By running commands like the following, it is easy to test run the network
training and see if the results are accurate. By adjusting the parameters, it is
possible to tweak and see how the efficiency and accuracy varies.

```
node xor.js --layers 2 --relu --value --epochs 1e5 --rate 0.01
```

Additionally, it is possible to graph the evolution of the total error during
the training, by using commands similar to below. Interestingly enough, it is
easy to see that sometimes - even with the same parameters - the network
converges while other times it ends up being chaotic.

```
# Set 1
for i in `seq 1 5`; do node xor.js --layers 2 --relu --value --epochs 1e5 --rate 0.01 --plot | sh plotter.sh > graph_$i.png; done
```

[Set 1](https://imgur.com/a/6aJGL)

```
# Set 2
for i in `seq 1 5`; do node xor.js --layers 2 --sigmoid --value --epochs 1e5 --rate 0.01 --plot | sh plotter.sh > graph_$i.png; done
```

[Set 2](https://imgur.com/a/0EDfI)

```
# Set 3
for i in `seq 1 5`; do node xor.js --layers 2 --relu --classification --epochs 1e5 --rate 0.01 --plot | sh plotter.sh > graph_$i.png; done
```

[Set 3](https://imgur.com/a/gBXNc)

```
# Set 4
for i in `seq 1 5`; do node xor.js --layers 2 --sigmoid --classification --epochs 1e5 --rate 0.01 --plot | sh plotter.sh > graph_$i.png; done
```

[Set 4](https://imgur.com/a/89bUG)

From above examples, it seems that at least with the current amount of epochs
and learning rate, sigmoid is rather unable to learn, and relu occasionally
learns.

Let's try more epochs.

```
# Set 5
for i in `seq 1 5`; do node xor.js --layers 2 --relu --value --epochs 1e6 --rate 0.01 --plot | sh plotter.sh > graph_$i.png; done;
```

[Set 5](https://imgur.com/a/kk5ki)

```
# Set 6
for i in `seq 1 5`; do node xor.js --layers 2 --sigmoid --value --epochs 1e6 --rate 0.01 --plot | sh plotter.sh > graph_$i.png; done;
```

[Set 6](https://imgur.com/a/vjmlx)

## Performance testing

TODO: Compare the runtimes with various number of layers and epochs

TODO: If there's extra time, implement a GPU veresion and compare to the naive
object version
