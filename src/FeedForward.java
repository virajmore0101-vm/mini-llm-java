import java.util.*;

/**
 * The feedforward sublayer: attention lets positions share information;
 * this is where the model actually THINKS about what it received, applied
 * independently to each position. ReLU(X*W1)*W2, plus a residual connection.
 * (Biases are skipped here to keep the backward pass simpler.)
 */
public class FeedForward {
    Matrix W1, W2;
    Param p1, p2;
    private Matrix X, preActivation, hidden;

    public FeedForward(int d, int hiddenSize, Random rand) {
        W1 = AttentionLayer.randomMatrix(d, hiddenSize, rand, 0.3);
        W2 = AttentionLayer.randomMatrix(hiddenSize, d, rand, 0.3);
        p1 = new Param(W1);
        p2 = new Param(W2);
    }

    /** A helper that shares the weights of `master` but has its own memory of the last forward pass and its own gradients. */
    FeedForward(FeedForward master) {
        W1 = new Matrix(master.W1.raw());
        W2 = new Matrix(master.W2.raw());
        p1 = new Param(W1);
        p2 = new Param(W2);
    }

    void collect(List<Param> out) { out.add(p1); out.add(p2); }

    public Matrix forward(Matrix X) {
        this.X = X;
        preActivation = X.multiply(W1);
        hidden = relu(preActivation);
        Matrix ffOutput = hidden.multiply(W2);
        return X.add(ffOutput);
    }

    /** Returns the gradient for the input and ADDS the weight gradients to p1 and p2 (no weight update here). */
    public Matrix backward(Matrix dLayerOutput) {
        Matrix dFFOutput = dLayerOutput;

        Matrix dHidden = dFFOutput.multiplyByTranspose(W2);
        p2.addGrad(hidden.transposeMultiply(dFFOutput));

        Matrix dPreActivation = reluBackward(dHidden, preActivation);
        p1.addGrad(X.transposeMultiply(dPreActivation));
        Matrix dX_fromFF = dPreActivation.multiplyByTranspose(W1);

        return dLayerOutput.add(dX_fromFF);
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
