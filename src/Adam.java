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

    /** Updates param in place. */
    public void step(double[] param, double[] grad, double lr) {
        if (PLAIN_SGD_FOR_GRADIENT_CHECK) {
            for (int i = 0; i < param.length; i++) param[i] -= lr * grad[i];
            return;
        }
        t++;
        double correction1 = 1 - Math.pow(BETA1, t);
        double correction2 = 1 - Math.pow(BETA2, t);
        for (int i = 0; i < param.length; i++) {
            m[i] = BETA1 * m[i] + (1 - BETA1) * grad[i];
            v[i] = BETA2 * v[i] + (1 - BETA2) * grad[i] * grad[i];
            param[i] -= lr * (m[i] / correction1) / (Math.sqrt(v[i] / correction2) + EPS);
        }
    }

    /** Same thing for a matrix; returns the updated matrix. */
    public Matrix step(Matrix param, Matrix grad, double lr) {
        int rows = param.rows(), cols = param.cols();
        double[] p = new double[rows * cols];
        double[] g = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                p[r * cols + c] = param.get(r, c);
                g[r * cols + c] = grad.get(r, c);
            }
        }
        step(p, g, lr);
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) result.set(r, c, p[r * cols + c]);
        }
        return result;
    }
}
