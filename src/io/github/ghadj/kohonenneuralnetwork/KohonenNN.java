package io.github.ghadj.kohonenneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class KohonenNN {
    private double[][][] weights;
    private double[][] distance;
    private char[][] labels;
    private int winnerX = 0;
    private int winnerY = 0;
    private double learningRate;
    private ArrayList<Double> trainErrorList = new ArrayList<Double>();
    private ArrayList<Double> testErrorList = new ArrayList<Double>();

    public KohonenNN(int gridSize, double learningRate, int dataDimension) {
        this.weights = new double[gridSize][gridSize][dataDimension];
        for (int i = 0; i < gridSize; i++)
            for (int j = 0; j < gridSize; j++)
                for (int k = 0; k < dataDimension; k++)
                    this.weights[i][j][k] = (new Random()).nextDouble() - 0.5;
        this.distance = new double[gridSize][gridSize];
        this.labels = new char[gridSize][gridSize];
        this.learningRate = learningRate;
    }

    public void run(Map<Character, List<Double>> data, Boolean train){

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