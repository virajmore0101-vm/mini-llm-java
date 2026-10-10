import java.util.*;

/**
 * The feedforward sublayer: attention lets positions share information;
 * this is where the model actually THINKS about what it received, applied
 * independently to each position. ReLU(X*W1)*W2, plus a residual connection.
 * (Biases are skipped here to keep the backward pass simpler.)
 */
public class FeedForward {
    Matrix W1, W2;
    private Adam adam1, adam2;
    private Matrix X, preActivation, hidden;

    public FeedForward(int d, int hiddenSize, Random rand) {
        W1 = AttentionLayer.randomMatrix(d, hiddenSize, rand, 0.3);
        W2 = AttentionLayer.randomMatrix(hiddenSize, d, rand, 0.3);
        adam1 = new Adam(d * hiddenSize);
        adam2 = new Adam(hiddenSize * d);
    }

    public Matrix forward(Matrix X) {
        this.X = X;
        preActivation = X.multiply(W1);
        hidden = relu(preActivation);
        Matrix ffOutput = hidden.multiply(W2);
        return X.add(ffOutput);
    }

    public Matrix backward(Matrix dLayerOutput, double learningRate) {
        Matrix dFFOutput = dLayerOutput;

        Matrix dHidden = dFFOutput.multiplyByTranspose(W2);
        Matrix dW2 = hidden.transposeMultiply(dFFOutput);

        Matrix dPreActivation = reluBackward(dHidden, preActivation);
        Matrix dW1 = X.transposeMultiply(dPreActivation);
        Matrix dX_fromFF = dPreActivation.multiplyByTranspose(W1);

        Matrix dX = dLayerOutput.add(dX_fromFF);

        W1 = adam1.step(W1, dW1, learningRate);
        W2 = adam2.step(W2, dW2, learningRate);

        return dX;
    }

    private static Matrix relu(Matrix m) {
        Matrix result = new Matrix(m.rows(), m.cols());
        for (int r = 0; r < m.rows(); r++)
            for (int c = 0; c < m.cols(); c++)
                result.set(r, c, Math.max(0, m.get(r, c)));
        return result;
    }

    private static Matrix reluBackward(Matrix dOut, Matrix preActivation) {
        Matrix result = new Matrix(dOut.rows(), dOut.cols());
        for (int r = 0; r < dOut.rows(); r++)
            for (int c = 0; c < dOut.cols(); c++)
                result.set(r, c, preActivation.get(r, c) > 0 ? dOut.get(r, c) : 0);
        return result;
    }
}
