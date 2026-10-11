/**
 * One block of learnable numbers: the weights themselves (updated in place, so they are the very numbers
 * the layer uses), the gradient collected since the last update, and Adam's memory for this block.
 * Gradients are ADDED to g, which is what lets several windows (or several threads) each contribute
 * before a single update is made.
 */
public class Param {
    final double[][] w;
    final double[][] g;
    private Adam adam;      // created on the first update; helper copies of a model never update, so they never need it

    Param(Matrix weights) { this(weights.raw()); }
    Param(double[][] weights) {
        w = weights;
        g = new double[w.length][w[0].length];
    }

    int size() { return w.length * w[0].length; }

    void zeroGrad() { for (double[] row : g) java.util.Arrays.fill(row, 0.0); }

    void addGrad(Matrix d) {
        for (int r = 0; r < g.length; r++) for (int c = 0; c < g[r].length; c++) g[r][c] += d.get(r, c);
    }

    /** For a vector parameter such as LayerNorm's gamma and beta, stored as a single row. */
    void addGrad(double[] d) { for (int c = 0; c < d.length; c++) g[0][c] += d[c]; }

    void addGradFrom(Param other) {
        for (int r = 0; r < g.length; r++) for (int c = 0; c < g[r].length; c++) g[r][c] += other.g[r][c];
    }

    void scaleGrad(double s) { for (double[] row : g) for (int c = 0; c < row.length; c++) row[c] *= s; }

    void copyWeightsFrom(Param other) { for (int r = 0; r < w.length; r++) System.arraycopy(other.w[r], 0, w[r], 0, w[r].length); }

    void step(double lr) {
        if (adam == null) adam = new Adam(size());
        adam.step(w, g, lr);
    }
}
