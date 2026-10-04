import java.util.*;

public class AttentionLayer {
    int d, numHeads, headDim;
    Matrix[] Wq, Wk, Wv;
    Matrix Wo;

    private Matrix X;
    private Matrix[] Q, K, V, attnWeights;
    private Matrix concatOutput;

    public AttentionLayer(int d, int numHeads, Random rand) {
        this.d = d;
        this.numHeads = numHeads;
        this.headDim = d / numHeads;
        Wq = new Matrix[numHeads];
        Wk = new Matrix[numHeads];
        Wv = new Matrix[numHeads];
        for (int h = 0; h < numHeads; h++) {
            Wq[h] = randomMatrix(d, headDim, rand, 0.5);
            Wk[h] = randomMatrix(d, headDim, rand, 0.5);
            Wv[h] = randomMatrix(d, headDim, rand, 0.5);
        }
        Wo = randomMatrix(d, d, rand, 0.5);
    }

    public Matrix forward(Matrix X) {
        this.X = X;
        int T = X.rows();
        Q = new Matrix[numHeads];
        K = new Matrix[numHeads];
        V = new Matrix[numHeads];
        attnWeights = new Matrix[numHeads];
        Matrix[] headOutputs = new Matrix[numHeads];

        for (int h = 0; h < numHeads; h++) {
            Q[h] = X.multiply(Wq[h]);
            K[h] = X.multiply(Wk[h]);
            V[h] = X.multiply(Wv[h]);
            Matrix scores = Q[h].multiply(K[h].transpose()).scale(1.0 / Math.sqrt(headDim));
            attnWeights[h] = softmaxRows(scores);
            headOutputs[h] = attnWeights[h].multiply(V[h]);
        }

        concatOutput = concat(headOutputs, T);
        Matrix attnOutput = concatOutput.multiply(Wo);
        return X.add(attnOutput);
    }

    public Matrix backward(Matrix dLayerOutput, double learningRate) {
        int T = X.rows();

        Matrix dConcatOutput = dLayerOutput.multiply(Wo.transpose());
        Matrix dWo = concatOutput.transpose().multiply(dLayerOutput);

        Matrix[] dHeadOutputs = splitByHead(dConcatOutput, T);
        Matrix dX = dLayerOutput;

        Matrix[] dWqArr = new Matrix[numHeads];
        Matrix[] dWkArr = new Matrix[numHeads];
        Matrix[] dWvArr = new Matrix[numHeads];

        for (int h = 0; h < numHeads; h++) {
            Matrix dV_h = attnWeights[h].transpose().multiply(dHeadOutputs[h]);
            Matrix dAttnWeights_h = dHeadOutputs[h].multiply(V[h].transpose());

            Matrix dScores_h = new Matrix(T, T);
            for (int i = 0; i < T; i++) {
                double dot = 0;
                for (int j = 0; j < T; j++) dot += attnWeights[h].get(i, j) * dAttnWeights_h.get(i, j);
                for (int j = 0; j < T; j++) {
                    dScores_h.set(i, j, attnWeights[h].get(i, j) * (dAttnWeights_h.get(i, j) - dot) / Math.sqrt(headDim));
                }
            }

            Matrix dQ_h = dScores_h.multiply(K[h]);
            Matrix dK_h = dScores_h.transpose().multiply(Q[h]);

            dWqArr[h] = X.transpose().multiply(dQ_h);
            dWkArr[h] = X.transpose().multiply(dK_h);
            dWvArr[h] = X.transpose().multiply(dV_h);

            Matrix dX_fromHead = dQ_h.multiply(Wq[h].transpose())
                .add(dK_h.multiply(Wk[h].transpose()))
                .add(dV_h.multiply(Wv[h].transpose()));

            dX = dX.add(dX_fromHead);
        }

        for (int h = 0; h < numHeads; h++) {
            Wq[h] = Wq[h].add(dWqArr[h].scale(-learningRate));
            Wk[h] = Wk[h].add(dWkArr[h].scale(-learningRate));
            Wv[h] = Wv[h].add(dWvArr[h].scale(-learningRate));
        }
        Wo = Wo.add(dWo.scale(-learningRate));

        return dX;
    }

    private Matrix concat(Matrix[] heads, int T) {
        Matrix result = new Matrix(T, d);
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < headDim; j++) {
                    result.set(i, h * headDim + j, heads[h].get(i, j));
                }
            }
        }
        return result;
    }

    private Matrix[] splitByHead(Matrix full, int T) {
        Matrix[] result = new Matrix[numHeads];
        for (int h = 0; h < numHeads; h++) {
            result[h] = new Matrix(T, headDim);
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < headDim; j++) {
                    result[h].set(i, j, full.get(i, h * headDim + j));
                }
            }
        }
        return result;
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
