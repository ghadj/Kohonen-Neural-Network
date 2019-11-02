package io.github.ghadj.kohonenneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class KohonenNN {
    private double[][][] weights;
    private char[][] labels;
    private int winnerX = 0;
    private int winnerY = 0;
    private double learningRate;
    private ArrayList<Double> trainErrorList = new ArrayList<Double>();
    private ArrayList<Double> testErrorList = new ArrayList<Double>();
    private double standardDeviation;
    private int T;
    private int gridSize;
    private int dataDimension;

    public KohonenNN(int gridSize, double learningRate, int maxIterations, int dataDimension,
            double standardDeviation) {
        this.weights = new double[gridSize][gridSize][dataDimension];
        this.gridSize = gridSize;
        this.dataDimension = dataDimension;

        for (int i = 0; i < gridSize; i++)
            for (int j = 0; j < gridSize; j++)
                for (int k = 0; k < dataDimension; k++)
                    this.weights[i][j][k] = (new Random()).nextDouble() - 0.5;

        this.labels = new char[gridSize][gridSize];
        this.learningRate = learningRate;
        this.standardDeviation = standardDeviation;
        this.T = maxIterations;
    }

    public void run(Map<Character, List<Double>> train, Map<Character, List<Double>> test) {
        for (int t = 0; t < this.T; t++) {
            epoch(t, train, true);
            epoch(t, test, false);
        }
        labeling(test);
        lvq(train);
        labeling(test);
    }

    private void epoch(int t, Map<Character, List<Double>> data, Boolean train) {
        double sumError = 0.0;
        int minX = 0, minY = 0;
        double dmin = Double.MAX_VALUE, d;
        for (Map.Entry<Character, List<Double>> l : data.entrySet()) {

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

    private void updateWeight(int x, int y, int z, int t, double in) {
        weights[x][y][z] = weights[x][y][z]
                + getCurrentLearningRate(t) * neighbourhoodFunction(x, y, t) * (in - weights[x][y][z]);
    }

    private double neighbourhoodFunction(int x, int y, int t) {
        return Math.exp((-1) * euclideanDist(x, y) / (2 * Math.pow(this.getCurrentStandardDeviation(t), 2)));
    }

    private double getCurrentLearningRate(int t) {
        return this.learningRate * Math.exp((-1) * ((double) t) / T);
    }

    private double getCurrentStandardDeviation(int t) {
        return this.standardDeviation * Math.exp((-1) * ((double) t) / (T / Math.log(this.standardDeviation)));
    }

    private double euclideanDist(int x, int y) {
        return Math.pow(winnerX - x, 2) + Math.pow(winnerY - y, 2);
    }

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

    public void lvq(Map<Character, List<Double>> data) {
        int minX = 0, minY = 0, sign;
        double dmin = Double.MAX_VALUE, d;

        for (Map.Entry<Character, List<Double>> l : data.entrySet()) {

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
                        + (sign) * (l.getValue().get(k) - weights[minX][minY][k]);
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

    public char[][] getLabels() {
        return labels;
    }
}