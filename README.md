# Kohonen Neural Network
Implementation of Kohonen neural network in java.

## Objectives
* help me to internalize the mathematical description of the algorithm.
* understand the algorithm intimately and discover parameter configurations.
* how the parameters of the algorithm influence its performance. 
* experiment with various datasets and see the behaviour of the algorithm.
* track performance of the algorithm-implementation with different metrics.
* light preprocessing of dataset.
* explore opportunities to make the implementation more efficient.

## Implementation
The following  parameters of the network can be set by a  text file, its path given as command line argument:
* grid size
* learning rate
* standard deviation
* data dimension
* number of iterations

## Compile & Run
```
javac -d ./bin ./src/io/github/ghadj/kohonenneuralnetwork/*.java

java -cp ./bin io.github.ghadj.kohonenneuralnetwork.KohonenNNDriver <path to parameters' file>
```
