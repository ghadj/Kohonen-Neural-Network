package io.github.ghadj.kohonenneuralnetwork;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Driver of the neural network. Takes as input from the arguments the path to
 * the file, containing the parameters of the neural network. Traings the neural
 * network based on the given parameters and exports the error and success rate
 * per epoch in two separate files.
 * 
 * Assume that the parameter file has the following format: gridSize 80
 * learningRate 0.3 maxIterations 10000 dataDimension 16 trainFile
 * ../Parameters/training.csv testFile ../Parameters/test.csv
 * 
 * Compile from Simple-Neural-Network/ directory javac -d ./bin
 * ./src/io/github/ghadj/simpleneuralnetwork/*.java
 * 
 * Run from Simple-Neural-Network/ directory java -cp ./bin
 * io.github.ghadj.simpleneuralnetwork.SimpleNeuralNetworkDriver <path to
 * parameters.txt>
 * 
 * @author Georgios Hadjiantonis
 * @since 01-11-2019
 */
public class KohonenNNDriver {
    private static final String errorFilename = "errors.txt";

    /**
     * Reads the parameters of the neural network from the given file.
     * 
     * @param filename path to the file containing the parameters
     * @return a String array containing the parameters.
     * @throws FileNotFoundException
     * @throws IOException
     * @throws InvalidParameterException
     */
    public static String[] readParameters(String filename)
            throws FileNotFoundException, IOException, InvalidParameterException {
        File file = new File(filename);
        BufferedReader br;
        String[] parameters = new String[7];
        int i = 0;

        br = new BufferedReader(new FileReader(file));
        String st;
        while ((st = br.readLine()) != null)
            parameters[i++] = st.split(" ")[1];
        br.close();

        if (i != 9)
            throw new InvalidParameterException("Invalid parameters given.");
        return parameters;
    }

    /**
     * Reads data from the given file. Returns a map in the form of <input list,
     * output list>.
     * 
     * @param numInputNeurons  number of input neurons.
     * @param numOutputNeurons number of output neurons.
     * @param filename         name of file to be read.
     * @return a map in the form of <input list, output list>.
     * @throws FileNotFoundException
     * @throws IOException
     * @throws InvalidParameterException in case the data of the given file is
     *                                   inconsistent.
     */
    public static Map<Character, List<Double>> readData(int dataDimension, String filename)
            throws FileNotFoundException, IOException, InvalidParameterException {
        Map<Character, List<Double>> data = new HashMap<Character, List<Double>>();
        File file = new File(filename);
        BufferedReader br;
        br = new BufferedReader(new FileReader(file));
        String st;
        while ((st = br.readLine()) != null) {
            List<Double> features = new ArrayList<>();
            int i = 0;
            String[] line = st.split(",");
            if (line.length != 1 + dataDimension) {
                br.close();
                throw new InvalidParameterException("Inconsistent data given in file " + filename);
            }

            char label = line[i++].charAt(0);
            for (int j = 0; j < dataDimension; j++)
                features.add(Double.parseDouble(line[i++]));

            data.put(label, features);
        }
        br.close();
        return data;
    }

    /**
     * Runs the NN based on the parameters, training and testing given data. Writes
     * the squares error and success rate to two separate files at the end of all
     * the iterations.
     * 
     * @param parameters
     * @param trainingData
     * @param testData
     * @throws IOException
     */
    public static void run(String[] parameters, Map<Character, List<Double>> trainingData,
            Map<Character, List<Double>> testData) throws IOException {

        KohonenNN nn = new KohonenNN(Integer.parseInt(parameters[0]), Double.parseDouble(parameters[1]), Integer.parseInt(parameters[2]),  Integer.parseInt(parameters[3]), Double.parseDouble(parameters[4]));
        //for (int i = 0; i < Integer.parseInt(parameters[6]); i++) {
        //    nn.run(trainingData, true);
        //   nn.run(testData, false);
        //}
        List<Double> trainError = nn.getTrainErrorList();
        List<Double> testError = nn.getTestErrorList();
        writeResults(trainError, testError, errorFilename);
    }

    /**
     * Writes the results in csv format to the file given.
     * 
     * @param trainResults
     * @param testResults
     * @param filename
     * @throws IOException
     */
    public static void writeResults(List<Double> trainResults, List<Double> testResults, String filename)
            throws IOException {
        Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < trainResults.size() && i < testResults.size(); i++)
            str.append((i + 1) + "," + trainResults.get(i) + "," + testResults.get(i) + "\n");
        writer.write(str.toString());
        writer.close();
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Error: Enter the path to the parameters.txt as an argument to the program.");
            return;
        }
        Map<Character, List<Double>> trainingData, testData;
        String[] parameters;
        try {
            parameters = readParameters(args[0]);
            trainingData = readData(Integer.parseInt(parameters[3]), parameters[5]);
            testData = readData(Integer.parseInt(parameters[3]), parameters[6]);

            run(parameters, trainingData, testData);
        } catch (InvalidParameterException e) {
            System.out.println("Error: " + e.getMessage());
        } catch (FileNotFoundException e) {
            System.out.println("Error: " + e.getMessage());
        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        }

    }
}