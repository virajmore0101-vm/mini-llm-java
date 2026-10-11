import java.util.*;

public class AttentionLayer {
    int d, numHeads, headDim;
    Matrix[] Wq, Wk, Wv;
    Matrix Wo;
    Param[] pQ, pK, pV;
    Param pO;

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
        makeParams();
    }

    /** A helper that shares the weights of `master` but has its own memory of the last forward pass and its own gradients. */
    AttentionLayer(AttentionLayer master) {
        d = master.d;
        numHeads = master.numHeads;
        headDim = master.headDim;
        Wq = new Matrix[numHeads];
        Wk = new Matrix[numHeads];
        Wv = new Matrix[numHeads];
        for (int h = 0; h < numHeads; h++) {
            Wq[h] = new Matrix(master.Wq[h].raw());
            Wk[h] = new Matrix(master.Wk[h].raw());
            Wv[h] = new Matrix(master.Wv[h].raw());
        }
        Wo = new Matrix(master.Wo.raw());
        makeParams();
    }

    private void makeParams() {
        pQ = new Param[numHeads];
        pK = new Param[numHeads];
        pV = new Param[numHeads];
        for (int h = 0; h < numHeads; h++) {
            pQ[h] = new Param(Wq[h]);
            pK[h] = new Param(Wk[h]);
            pV[h] = new Param(Wv[h]);
        }
        pO = new Param(Wo);
    }

    void collect(List<Param> out) {
        for (int h = 0; h < numHeads; h++) { out.add(pQ[h]); out.add(pK[h]); out.add(pV[h]); }
        out.add(pO);
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
            Matrix scores = Q[h].multiplyByTranspose(K[h]).scale(1.0 / Math.sqrt(headDim));
            // Causal mask: position i may only look at positions 0..i (never at the future).
            for (int i = 0; i < T; i++) for (int j = i + 1; j < T; j++) scores.set(i, j, Double.NEGATIVE_INFINITY);
            attnWeights[h] = softmaxRows(scores);
            headOutputs[h] = attnWeights[h].multiply(V[h]);
        }

        concatOutput = concat(headOutputs, T);
        Matrix attnOutput = concatOutput.multiply(Wo);
        return X.add(attnOutput);
    }

    /** Returns the gradient for the input and ADDS the weight gradients to the Params (no weight update here). */
    public Matrix backward(Matrix dLayerOutput) {
        int T = X.rows();

        Matrix dConcatOutput = dLayerOutput.multiplyByTranspose(Wo);
        pO.addGrad(concatOutput.transposeMultiply(dLayerOutput));

        Matrix[] dHeadOutputs = splitByHead(dConcatOutput, T);
        Matrix dX = dLayerOutput;

        for (int h = 0; h < numHeads; h++) {
            Matrix dV_h = attnWeights[h].transposeMultiply(dHeadOutputs[h]);
            Matrix dAttnWeights_h = dHeadOutputs[h].multiplyByTranspose(V[h]);

            Matrix dScores_h = new Matrix(T, T);
            for (int i = 0; i < T; i++) {
                double dot = 0;
                for (int j = 0; j < T; j++) dot += attnWeights[h].get(i, j) * dAttnWeights_h.get(i, j);
                for (int j = 0; j < T; j++) {
                    dScores_h.set(i, j, attnWeights[h].get(i, j) * (dAttnWeights_h.get(i, j) - dot) / Math.sqrt(headDim));
                }
            }

            Matrix dQ_h = dScores_h.multiply(K[h]);
            Matrix dK_h = dScores_h.transposeMultiply(Q[h]);

            pQ[h].addGrad(X.transposeMultiply(dQ_h));
            pK[h].addGrad(X.transposeMultiply(dK_h));
            pV[h].addGrad(X.transposeMultiply(dV_h));

            Matrix dX_fromHead = dQ_h.multiplyByTranspose(Wq[h])
                .add(dK_h.multiplyByTranspose(Wk[h]))
                .add(dV_h.multiplyByTranspose(Wv[h]));

            dX = dX.add(dX_fromHead);
        }
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
