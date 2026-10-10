import java.util.*;
import java.util.function.DoubleSupplier;

/** Proves two things: (1) position i never sees the future, (2) the backward pass matches numerical derivatives. */
public class GradCheck {
    static Random pick = new Random(1);
    static int checked = 0, bad = 0;
    static double worst = 0;
    static double maxGrad = 0;

    static void cmp(String name, double analytic, double numeric) {
        double diff = Math.abs(analytic - numeric), size = Math.abs(analytic) + Math.abs(numeric);
        checked++;
        maxGrad = Math.max(maxGrad, Math.abs(analytic));
        if (size > 1e-5) worst = Math.max(worst, diff / size);
        if (diff > 1e-6 && diff / size > 1e-3) { bad++; System.out.printf("  MISMATCH %s analytic=%.3e numeric=%.3e%n", name, analytic, numeric); }
    }

    static Matrix copy(Matrix m) {
        Matrix c = new Matrix(m.rows(), m.cols());
        for (int r = 0; r < m.rows(); r++) for (int k = 0; k < m.cols(); k++) c.set(r, k, m.get(r, k));
        return c;
    }

    static void checkMatrix(String name, Matrix before, Matrix after, double lr, DoubleSupplier loss, Matrix live, int n) {
        for (int t = 0; t < n; t++) {
            int r = pick.nextInt(before.rows()), c = pick.nextInt(before.cols());
            double analytic = (before.get(r, c) - after.get(r, c)) / lr;   // one plain-SGD step moved the weight by -lr*gradient
            double orig = live.get(r, c), eps = 1e-5;
            live.set(r, c, orig + eps); double up = loss.getAsDouble();
            live.set(r, c, orig - eps); double dn = loss.getAsDouble();
            live.set(r, c, orig);
            cmp(name + "[" + r + "," + c + "]", analytic, (up - dn) / (2 * eps));
        }
    }

    public static void main(String[] args) throws Exception {
        String text = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get("shakespeare.txt"))).substring(0, 20000);
        int T = 12, D = 24, L = 3, H = 4;
        String window = text.substring(500, 500 + T + 1);

        // 1) causality: changing a character must not change the hidden state of any EARLIER position
        MultiLayerTransformer m = new MultiLayerTransformer(text, T, D, L, H, 0.001);
        String input = window.substring(0, T);
        Matrix base = m.hiddenStates(input);
        double leak = 0;
        for (int j = 1; j < T; j++) {
            char other = input.charAt(j) == 'e' ? 't' : 'e';
            Matrix changed = m.hiddenStates(input.substring(0, j) + other + input.substring(j + 1));
            for (int i = 0; i < j; i++) for (int k = 0; k < D; k++) leak = Math.max(leak, Math.abs(base.get(i, k) - changed.get(i, k)));
        }
        System.out.printf("CAUSALITY: largest change in an earlier position when a later character changes = %.1e %s%n", leak, leak < 1e-12 ? "(OK)" : "(FAIL)");

        // 2) gradients: analytic (from one tiny plain-SGD step) vs numerical (finite differences)
        Adam.PLAIN_SGD_FOR_GRADIENT_CHECK = true;
        MultiLayerTransformer A = new MultiLayerTransformer(text, T, D, L, H, 0.001);
        MultiLayerTransformer B = new MultiLayerTransformer(text, T, D, L, H, 0.001);
        double lr = 1e-3;
        Matrix emb0 = copy(B.embeddingTable), out0 = copy(B.Wout);
        Matrix[][] q0 = new Matrix[L][H], k0 = new Matrix[L][H], v0 = new Matrix[L][H];
        Matrix[] o0 = new Matrix[L], w10 = new Matrix[L], w20 = new Matrix[L];
        double[][] g0 = new double[2 * L][], b0 = new double[2 * L][];
        for (int l = 0; l < L; l++) {
            for (int h = 0; h < H; h++) { q0[l][h] = copy(B.layers[l].Wq[h]); k0[l][h] = copy(B.layers[l].Wk[h]); v0[l][h] = copy(B.layers[l].Wv[h]); }
            o0[l] = copy(B.layers[l].Wo); w10[l] = copy(B.feedforwards[l].W1); w20[l] = copy(B.feedforwards[l].W2);
            g0[l] = B.attnNorms[l].gamma.clone(); b0[l] = B.attnNorms[l].beta.clone();
            g0[L + l] = B.ffNorms[l].gamma.clone(); b0[L + l] = B.ffNorms[l].beta.clone();
        }
        B.trainSequence(window);
        DoubleSupplier loss = () -> A.sequenceLoss(window);

        checkMatrix("embedding", emb0, B.embeddingTable, lr, loss, A.embeddingTable, 12);
        checkMatrix("Wout", out0, B.Wout, lr, loss, A.Wout, 12);
        for (int l = 0; l < L; l++) {
            for (int h = 0; h < H; h++) {
                checkMatrix("L" + l + ".Wq" + h, q0[l][h], B.layers[l].Wq[h], lr, loss, A.layers[l].Wq[h], 3);
                checkMatrix("L" + l + ".Wk" + h, k0[l][h], B.layers[l].Wk[h], lr, loss, A.layers[l].Wk[h], 3);
                checkMatrix("L" + l + ".Wv" + h, v0[l][h], B.layers[l].Wv[h], lr, loss, A.layers[l].Wv[h], 3);
            }
            checkMatrix("L" + l + ".Wo", o0[l], B.layers[l].Wo, lr, loss, A.layers[l].Wo, 5);
            checkMatrix("L" + l + ".W1", w10[l], B.feedforwards[l].W1, lr, loss, A.feedforwards[l].W1, 5);
            checkMatrix("L" + l + ".W2", w20[l], B.feedforwards[l].W2, lr, loss, A.feedforwards[l].W2, 5);
            for (int which = 0; which < 2; which++) {
                LayerNorm la = which == 0 ? A.attnNorms[l] : A.ffNorms[l], lb = which == 0 ? B.attnNorms[l] : B.ffNorms[l];
                int idx = which * L + l;
                String tag = "L" + l + (which == 0 ? ".attnNorm" : ".ffNorm");
                for (int t = 0; t < 3; t++) {
                    int j = pick.nextInt(D);
                    double eps = 1e-5, og = la.gamma[j], ob = la.beta[j];
                    la.gamma[j] = og + eps; double up = loss.getAsDouble(); la.gamma[j] = og - eps; double dn = loss.getAsDouble(); la.gamma[j] = og;
                    cmp(tag + ".gamma[" + j + "]", (g0[idx][j] - lb.gamma[j]) / lr, (up - dn) / (2 * eps));
                    la.beta[j] = ob + eps; up = loss.getAsDouble(); la.beta[j] = ob - eps; dn = loss.getAsDouble(); la.beta[j] = ob;
                    cmp(tag + ".beta[" + j + "]", (b0[idx][j] - lb.beta[j]) / lr, (up - dn) / (2 * eps));
                }
            }
        }
        Adam.PLAIN_SGD_FOR_GRADIENT_CHECK = false;
        System.out.printf("GRADIENTS: %d weights checked (largest gradient %.1e), %d mismatches, worst relative error %.1e %s%n",
            checked, maxGrad, bad, worst, bad == 0 ? "(OK)" : "(FAIL)");
    }
}
