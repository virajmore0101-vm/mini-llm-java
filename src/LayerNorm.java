import java.util.List;

public class LayerNorm {
    double[] gamma, beta;
    Param pGamma, pBeta;
    private Matrix X;
    private Matrix normalized;
    private double[] mean, variance;
    private static final double EPS = 1e-5;

    public LayerNorm(int d) {
        gamma = new double[d];
        beta = new double[d];
        for (int j = 0; j < d; j++) { gamma[j] = 1.0; beta[j] = 0.0; }
        pGamma = new Param(new double[][] { gamma });   // a one-row view of the same numbers, updated in place
        pBeta = new Param(new double[][] { beta });
    }

    /** A helper that shares gamma and beta with `master` but has its own memory of the last forward pass and its own gradients. */
    LayerNorm(LayerNorm master) {
        gamma = master.gamma;
        beta = master.beta;
        pGamma = new Param(new double[][] { gamma });
        pBeta = new Param(new double[][] { beta });
    }

    void collect(List<Param> out) { out.add(pGamma); out.add(pBeta); }

    public Matrix forward(Matrix X) {
        this.X = X;
        int T = X.rows(), d = X.cols();
        normalized = new Matrix(T, d);
        mean = new double[T];
        variance = new double[T];
        Matrix output = new Matrix(T, d);

        for (int i = 0; i < T; i++) {
            double m = 0;
            for (int j = 0; j < d; j++) m += X.get(i, j);
            m /= d;
            double v = 0;
            for (int j = 0; j < d; j++) v += (X.get(i, j) - m) * (X.get(i, j) - m);
            v /= d;
            mean[i] = m;
            variance[i] = v;
            double std = Math.sqrt(v + EPS);
            for (int j = 0; j < d; j++) {
                double norm = (X.get(i, j) - m) / std;
                normalized.set(i, j, norm);
                output.set(i, j, gamma[j] * norm + beta[j]);
            }
        }
        return output;
    }

    /** Returns the gradient for the input and ADDS the gradients of gamma and beta to their Params (no weight update here). */
    public Matrix backward(Matrix dOut) {
        int T = X.rows(), d = X.cols();
        Matrix dX = new Matrix(T, d);
        double[] dGamma = new double[d];
        double[] dBeta = new double[d];

        for (int i = 0; i < T; i++) {
            double std = Math.sqrt(variance[i] + EPS);
            double[] dNorm = new double[d];
            for (int j = 0; j < d; j++) {
                dNorm[j] = dOut.get(i, j) * gamma[j];
                dGamma[j] += dOut.get(i, j) * normalized.get(i, j);
                dBeta[j] += dOut.get(i, j);
            }
            double dNormSum = 0, dNormDotNorm = 0;
            for (int j = 0; j < d; j++) {
                dNormSum += dNorm[j];
                dNormDotNorm += dNorm[j] * normalized.get(i, j);
            }
            for (int j = 0; j < d; j++) {
                double val = (d * dNorm[j] - dNormSum - normalized.get(i, j) * dNormDotNorm) / (d * std);
                dX.set(i, j, val);
            }
        }

        pGamma.addGrad(dGamma);
        pBeta.addGrad(dBeta);
        return dX;
    }
}
