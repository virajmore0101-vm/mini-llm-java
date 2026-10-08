import java.util.Random;

public class NeuralNetwork {

    private final Matrix weightsInputHidden;
    private final Vector biasHidden;
    private final Matrix weightsHiddenOutput;
    private final Vector biasOutput;
    private final double learningRate;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.learningRate = learningRate;
        Random rand = new Random();

        weightsInputHidden = randomMatrix(hiddenSize, inputSize, rand);
        biasHidden = randomVector(hiddenSize, rand);
        weightsHiddenOutput = randomMatrix(outputSize, hiddenSize, rand);
        biasOutput = randomVector(outputSize, rand);
    }

    private Matrix randomMatrix(int rows, int cols, Random rand) {
        Matrix m = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                m.set(r, c, (rand.nextDouble() * 2 - 1) * 0.5);
            }
        }
        return m;
    }

    private Vector randomVector(int size, Random rand) {
        Vector v = new Vector(size);
        for (int i = 0; i < size; i++) {
            v.set(i, (rand.nextDouble() * 2 - 1) * 0.5);
        }
        return v;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private Vector sigmoid(Vector v) {
        Vector result = new Vector(v.size());
        for (int i = 0; i < v.size(); i++) {
            result.set(i, sigmoid(v.get(i)));
        }
        return result;
    }

    private Vector sigmoidDerivative(Vector sigmoidOutput) {
        Vector result = new Vector(sigmoidOutput.size());
        for (int i = 0; i < sigmoidOutput.size(); i++) {
            double s = sigmoidOutput.get(i);
            result.set(i, s * (1 - s));
        }
        return result;
    }

    public Vector predict(Vector input) {
        Vector hidden = sigmoid(weightsInputHidden.multiply(input).add(biasHidden));
        return sigmoid(weightsHiddenOutput.multiply(hidden).add(biasOutput));
    }

    public void trainStep(Vector input, Vector target) {
        Vector hidden = sigmoid(weightsInputHidden.multiply(input).add(biasHidden));
        Vector output = sigmoid(weightsHiddenOutput.multiply(hidden).add(biasOutput));

        Vector outputError = target.subtract(output);
        Vector outputDelta = outputError.multiplyElementwise(sigmoidDerivative(output));

        Vector hiddenError = weightsHiddenOutput.transpose().multiply(outputDelta);
        Vector hiddenDelta = hiddenError.multiplyElementwise(sigmoidDerivative(hidden));

        addInPlace(weightsHiddenOutput, Matrix.outerProduct(outputDelta, hidden).scale(learningRate));
        addInPlace(biasOutput, outputDelta.scale(learningRate));

        addInPlace(weightsInputHidden, Matrix.outerProduct(hiddenDelta, input).scale(learningRate));
        addInPlace(biasHidden, hiddenDelta.scale(learningRate));
    }

    private void addInPlace(Matrix target, Matrix delta) {
        for (int r = 0; r < target.rows(); r++) {
            for (int c = 0; c < target.cols(); c++) {
                target.set(r, c, target.get(r, c) + delta.get(r, c));
            }
        }
    }

    private void addInPlace(Vector target, Vector delta) {
        for (int i = 0; i < target.size(); i++) {
            target.set(i, target.get(i) + delta.get(i));
        }
    }
}
