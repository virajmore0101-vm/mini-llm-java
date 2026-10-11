/**
 * Adam optimizer for one block of parameters (one weight matrix or vector).
 * Plain gradient descent moves every weight by the same recipe: learning rate x gradient.
 * Adam keeps two running averages per weight: the gradient itself (momentum, smooths out noise)
 * and the squared gradient (how big this weight's gradients usually are). Each weight's step is
 * then scaled to its own typical size, so quiet weights still learn and jumpy ones calm down.
 */
public class Adam {
    private static final double BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;
    private final double[] m, v;
    private int t = 0;

    /** Only GradCheck turns this on: it swaps Adam for plain gradient descent so the raw gradient can be read back. */
    public static boolean PLAIN_SGD_FOR_GRADIENT_CHECK = false;

    public Adam(int size) {
        m = new double[size];
        v = new double[size];
    }

    /** Updates the weights w in place. w and g have the same shape; m and v are indexed row by row. */
    public void step(double[][] w, double[][] g, double lr) {
        if (PLAIN_SGD_FOR_GRADIENT_CHECK) {
            for (int r = 0; r < w.length; r++) for (int c = 0; c < w[r].length; c++) w[r][c] -= lr * g[r][c];
            return;
        }
        t++;
        double correction1 = 1 - Math.pow(BETA1, t);
        double correction2 = 1 - Math.pow(BETA2, t);
        int i = 0;
        for (int r = 0; r < w.length; r++) {
            double[] wr = w[r], gr = g[r];
            for (int c = 0; c < wr.length; c++, i++) {
                m[i] = BETA1 * m[i] + (1 - BETA1) * gr[c];
                v[i] = BETA2 * v[i] + (1 - BETA2) * gr[c] * gr[c];
                wr[c] -= lr * (m[i] / correction1) / (Math.sqrt(v[i] / correction2) + EPS);
            }
        }
    }
}
