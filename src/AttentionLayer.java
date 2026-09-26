import java.util.*;

public class AttentionLayer {
    Matrix Wq, Wk, Wv;
    int d;
    private Matrix X, Q, K, V, attnWeights;

    public AttentionLayer(int d, Random rand) {
        this.d = d;
        Wq = randomMatrix(d, d, rand, 0.5);
        Wk = randomMatrix(d, d, rand, 0.5);
        Wv = randomMatrix(d, d, rand, 0.5);
    }

    public Matrix forward(Matrix X) {
        this.X = X;
        Q = X.multiply(Wq);
        K = X.multiply(Wk);
        V = X.multiply(Wv);
        Matrix scores = Q.multiply(K.transpose()).scale(1.0 / Math.sqrt(d));
        attnWeights = softmaxRows(scores);
        Matrix attnOutput = attnWeights.multiply(V);
        return X.add(attnOutput);
    }

    public Matrix backward(Matrix dLayerOutput, double learningRate) {
        int T = X.rows();

        Matrix dV = attnWeights.transpose().multiply(dLayerOutput);
        Matrix dAttnWeights = dLayerOutput.multiply(V.transpose());

        Matrix dScores = new Matrix(T, T);
        for (int i = 0; i < T; i++) {
            double dot = 0;
            for (int j = 0; j < T; j++) dot += attnWeights.get(i, j) * dAttnWeights.get(i, j);
            for (int j = 0; j < T; j++) {
                dScores.set(i, j, attnWeights.get(i, j) * (dAttnWeights.get(i, j) - dot) / Math.sqrt(d));
            }
        }

        Matrix dQ = dScores.multiply(K);
        Matrix dK = dScores.transpose().multiply(Q);

        Matrix dWq = X.transpose().multiply(dQ);
        Matrix dWk = X.transpose().multiply(dK);
        Matrix dWv = X.transpose().multiply(dV);

        Matrix dX = dLayerOutput
            .add(dQ.multiply(Wq.transpose()))
            .add(dK.multiply(Wk.transpose()))
            .add(dV.multiply(Wv.transpose()));

        Wq = Wq.add(dWq.scale(-learningRate));
        Wk = Wk.add(dWk.scale(-learningRate));
        Wv = Wv.add(dWv.scale(-learningRate));

        return dX;
    }

    static Matrix softmaxRows(Matrix scores) {
        Matrix result = new Matrix(scores.rows(), scores.cols());
        for (int r = 0; r < scores.rows(); r++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < scores.cols(); c++) max = Math.max(max, scores.get(r, c));
            double sum = 0;
            double[] exps = new double[scores.cols()];
            for (int c = 0; c < scores.cols(); c++) { exps[c] = Math.exp(scores.get(r, c) - max); sum += exps[c]; }
            for (int c = 0; c < scores.cols(); c++) result.set(r, c, exps[c] / sum);
        }
        return result;
    }

    static Matrix randomMatrix(int rows, int cols, Random rand, double scale) {
        Matrix m = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) m.set(r, c, (rand.nextDouble() * 2 - 1) * scale);
        return m;
    }
}
