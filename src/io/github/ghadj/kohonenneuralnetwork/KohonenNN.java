package io.github.ghadj.kohonenneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Implementation of Kohonen Neural Network.
 * 
 * @author Georgios Hadjiantonis
 * @since 02-11-2019
 */
public class KohonenNN {
    private double[][][] weights; // weight from each neuron to each input feature
    private char[][] labels; // char corresponding to each neuron
    private int winnerX = 0; // x coordinate of winner neuron
    private int winnerY = 0; // y coordinate of winner neuron
    private double learningRate;
    private ArrayList<Double> trainErrorList = new ArrayList<Double>();
    private ArrayList<Double> testErrorList = new ArrayList<Double>();
    private double standardDeviation;
    private int T; // number of total iterations
    private int gridSize; // size of grid
    private int dataDimension; // number of features per input

    /**
     * Constructor of Neural Network. Initializes weights to random double [-1, 1],
     * and sets attributes to the given values.
     * 
     * @param gridSize          size of the grid.
     * @param learningRate      learning rate.
     * @param maxIterations     number of epoches to be executed.
     * @param dataDimension     dimension of the input data, number of features per
     *                          input.
     * @param standardDeviation standard deviation of the Gaussian function.
     */
    public KohonenNN(int gridSize, double learningRate, int maxIterations, int dataDimension,
            double standardDeviation) {
        this.weights = new double[gridSize][gridSize][dataDimension];
        this.gridSize = gridSize;
        this.dataDimension = dataDimension;

        for (int i = 0; i < gridSize; i++)
            for (int j = 0; j < gridSize; j++)
                for (int k = 0; k < dataDimension; k++)
                    this.weights[i][j][k] = (new Random()).nextDouble() * 2 - 1; // [-1, 1]

        this.labels = new char[gridSize][gridSize];
        this.learningRate = learningRate;
        this.standardDeviation = standardDeviation;
        this.T = maxIterations;
    }

    /**
     * Runs the neural network, each iteration one epoch with training and another
     * with test data. At the end, labels neurons (using test data), applies LVQ
     * (using training data) and then labels again.
     * 
     * @param train data.
     * @param test  data.
     */
    public void run(Map<Character, List<Double>> train, Map<Character, List<Double>> test) {
        for (int t = 0; t < this.T; t++) {
            System.out.println("Running Epoch: #" + t);
            epoch(t, train, true);
            epoch(t, test, false);
        }
        labeling(test);
        lvq(train);
        labeling(test);
    }

    /**
     * Runs an epoch, through all the data given. If train then updates weights
     * after each pattern. Calculates training/testing error for this epoch.
     * 
     * @param t     current epoch.
     * @param data.
     * @param train true if training, otherwise false.
     */
    private void epoch(int t, Map<Character, List<Double>> data, Boolean train) {
        double sumError = 0.0;
        int minX = 0, minY = 0;
        double dmin, d;
        for (Map.Entry<Character, List<Double>> l : data.entrySet()) {
            dmin = Double.MAX_VALUE;
            for (int i = 0; i < this.gridSize; i++)
                for (int j = 0; j < this.gridSize; j++) {
                    d = 0;
                    for (int k = 0; k < this.dataDimension; k++)
                        d += Math.pow(l.getValue().get(k) - weights[i][j][k], 2);

                    if (d < dmin) {
                        minX = i;
                        minY = j;
                        dmin = d;
                    }
                }

            if (train) {
                this.winnerX = minX;
                this.winnerY = minY;

                for (int i = 0; i < this.gridSize; i++)
                    for (int j = 0; j < this.gridSize; j++)
                        for (int k = 0; k < this.dataDimension; k++)
                            updateWeight(i, j, k, t, l.getValue().get(k));
            }
            sumError += dmin;
        }

        if (train)
            this.trainErrorList.add(sumError / data.size());
        else
            this.testErrorList.add(sumError / data.size());
    }

    /**
     * Updates weight of the given neuron, specified by coordinates x and y.
     * 
     * @param x  coordinate of neuron.
     * @param y  coordinate of neuron.
     * @param z  weight to input feature.
     * @param t  current epoch.
     * @param in input feature.
     */
    private void updateWeight(int x, int y, int z, int t, double in) {
        weights[x][y][z] = weights[x][y][z]
                + getCurrentLearningRate(t) * neighbourhoodFunction(x, y, t) * (in - weights[x][y][z]);
    }

    /**
     * Returns value of neighbourhood function.
     * 
     * @param x coordinate of neuron.
     * @param y coordinate of neuron.
     * @param t current epoch.
     * @return value of neighbourhood function.
     */
    private double neighbourhoodFunction(int x, int y, int t) {
        return Math.exp((-1) * euclideanDist(x, y) / (2 * Math.pow(this.getCurrentStandardDeviation(t), 2)));
    }

    /**
     * Returns current learning rate (decreases over the iterations).
     * 
     * @param t current epoch.
     * @return current learning rate.
     */
    private double getCurrentLearningRate(int t) {
        return this.learningRate * Math.exp((-1) * ((double) t) / T);
    }

    /**
     * Returns current standard deviation of the Gaussian function (decreases over
     * the iterations).
     * 
     * @param t current epoch.
     * @return current standard deviation.
     */
    private double getCurrentStandardDeviation(int t) {
        return this.standardDeviation
                * Math.exp((-1) * ((double) t) / (((double) T) / Math.log10(this.standardDeviation)));
    }

    /**
     * Returns the euclidan distance between the neuron given, represented by the
     * coordinates x and y, and the winner neuron.
     * 
     * @param x coordinate of neuron.
     * @param y coordinate of neuron.
     * @return euclidan distance between the neuron given and the winner neuron.
     */
    private double euclideanDist(int x, int y) {
        return Math.pow(winnerX - x, 2) + Math.pow(winnerY - y, 2);
    }

    /**
     * Labels neurons based on the input with the shortest distance.
     * 
     * @param data testing data.
     */
    public void labeling(Map<Character, List<Double>> data) {
        double dmin, d;
        char cmin;
        for (int i = 0; i < this.gridSize; i++)
            for (int j = 0; j < this.gridSize; j++) {
                dmin = Double.MAX_VALUE;
                cmin = 0;
                for (Map.Entry<Character, List<Double>> l : data.entrySet()) {
                    d = 0;
                    for (int k = 0; k < this.dataDimension; k++)
                        d += Math.pow(l.getValue().get(k) - weights[i][j][k], 2);

                    if (d < dmin) {
                        cmin = l.getKey();
                        dmin = d;
                    }
                }
                labels[i][j] = cmin;
            }
    }

    /**
     * Applies Learning Vector Quantification.
     * 
     * @param data training data.
     */
    public void lvq(Map<Character, List<Double>> data) {
        int minX = 0, minY = 0, sign;
        double dmin, d;

        for (Map.Entry<Character, List<Double>> l : data.entrySet()) {
            dmin = Double.MAX_VALUE;
            for (int i = 0; i < this.gridSize; i++)
                for (int j = 0; j < this.gridSize; j++) {
                    d = 0;
                    for (int k = 0; k < this.dataDimension; k++)
                        d += Math.pow(l.getValue().get(k) - weights[i][j][k], 2);

                    if (d < dmin) {
                        minX = i;
                        minY = j;
                        dmin = d;
                    }
                }

            sign = (labels[minX][minY] == l.getKey()) ? 1 : -1;
            for (int k = 0; k < this.dataDimension; k++)
                weights[minX][minY][k] = weights[minX][minY][k]
                        + (sign) * this.getCurrentLearningRate(this.T) * (l.getValue().get(k) - weights[minX][minY][k]);
        }

    }

    /**
     * Returns a list containing the mean of the squared error per epoch, during
     * training.
     * 
     * @return list containing the mean of the squared error per epoch.
     */
    public ArrayList<Double> getTrainErrorList() {
        return trainErrorList;
    }

    /**
     * Returns a list containing the mean of the squared error per epoch, during
     * test.
     * 
     * @return list containing the mean of the squared error per epoch.
     */
    public ArrayList<Double> getTestErrorList() {
        return testErrorList;
    }

    /**
     * Returns array of characters containing the label-character per neuron.
     * 
     * @return array of characters containing the label-character per neuron.
     */
    public char[][] getLabels() {
        return labels;
    }
}