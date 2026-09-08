import java.util.Random;

/**
 * A single trainable neuron: weights + bias + activation function,
 * updated via gradient descent.
 */
public class Neuron {

    private final Vector weights;
    private double bias;
    private final double learningRate;

    public Neuron(int numInputs, double learningRate) {
        Random rand = new Random();
        double[] initial = new double[numInputs];
        for (int i = 0; i < numInputs; i++) {
            initial[i] = (rand.nextDouble() * 2 - 1) * 0.5;
        }
        this.weights = new Vector(initial);
        this.bias = (rand.nextDouble() * 2 - 1) * 0.5;
        this.learningRate = learningRate;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double predict(Vector input) {
        double weightedSum = weights.dot(input) + bias;
        return sigmoid(weightedSum);
    }

    public void trainStep(Vector input, double target) {
        double prediction = predict(input);
        double error = target - prediction;
        double gradient = error * prediction * (1 - prediction);

        for (int i = 0; i < weights.size(); i++) {
            double updated = weights.get(i) + learningRate * gradient * input.get(i);
            weights.set(i, updated);
        }
        bias += learningRate * gradient;
    }

    public Vector getWeights() { return weights; }
    public double getBias() { return bias; }
}
