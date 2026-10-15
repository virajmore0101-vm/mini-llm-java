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

    static MultiLayerTransformer make(Tokenizer tok, String text, int T, int D, int L, int H) {
        return tok == null ? new MultiLayerTransformer(text, T, D, L, H, 0.001) : new MultiLayerTransformer(tok, T, D, L, H, 0.001);
    }

    public static void main(String[] args) throws Exception {
        String text = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get("shakespeare.txt"))).substring(0, 20000);
        run("CHARACTER-LEVEL MODEL", text, null);
        checked = 0; bad = 0; worst = 0; maxGrad = 0;
        run("WORD-PIECE (BPE) MODEL, 300 tokens", text, Tokenizer.train(text, 300));
    }

    static void run(String label, String text, Tokenizer tok) {
        System.out.println("--- " + label + " ---");
        int T = 12, D = 24, L = 3, H = 4;
        Tokenizer coder = tok != null ? tok : Tokenizer.chars(text);
        int[] allIds = coder.encode(text);
        int[] window = Arrays.copyOfRange(allIds, 500, 500 + T + 1);          // T+1 tokens

        // 1) causality: changing a character must not change the hidden state of any EARLIER position
        MultiLayerTransformer m = make(tok, text, T, D, L, H);
        int[] input = Arrays.copyOf(window, T);
        Matrix base = m.hiddenStatesIds(input);
        double leak = 0;
        for (int j = 1; j < T; j++) {
            int[] altered = input.clone();
            altered[j] = (input[j] + 1) % m.vocabSize;
            Matrix changed = m.hiddenStatesIds(altered);
            for (int i = 0; i < j; i++) for (int k = 0; k < D; k++) leak = Math.max(leak, Math.abs(base.get(i, k) - changed.get(i, k)));
        }
        System.out.printf("CAUSALITY: largest change in an earlier position when a later token changes = %.1e %s%n", leak, leak < 1e-12 ? "(OK)" : "(FAIL)");

        // 2) gradients: analytic (from one tiny plain-SGD step) vs numerical (finite differences)
        Adam.PLAIN_SGD_FOR_GRADIENT_CHECK = true;
        MultiLayerTransformer A = make(tok, text, T, D, L, H);
        MultiLayerTransformer B = make(tok, text, T, D, L, H);
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

        // 3) mini-batches and threads: the averaged gradient of a batch computed by several threads must be the true derivative
        //    of the batch's average loss, and must match what a single thread computes.
        int NB = 8, smallerStep = 0;
        int[][] batch = new int[NB][];
        for (int i = 0; i < NB; i++) batch[i] = Arrays.copyOfRange(allIds, 2000 + 311 * i, 2000 + 311 * i + T + 1);
        int before = checked, badBefore = bad;
        double worstBefore = worst;
        worst = 0; maxGrad = 0;
        Adam.PLAIN_SGD_FOR_GRADIENT_CHECK = true;
        MultiLayerTransformer live = make(tok, text, T, D, L, H);   // used only to measure the loss
        MultiLayerTransformer single = make(tok, text, T, D, L, H);
        MultiLayerTransformer multi = make(tok, text, T, D, L, H);
        multi.setThreads(3);
        double[][][] w0 = snapshot(multi);
        double lossSingle = single.trainBatch(batch), lossMulti = multi.trainBatch(batch);
        DoubleSupplier batchLoss = () -> live.batchLoss(batch);
        for (int i = 0; i < multi.params.size(); i++) {
            double[][] w = live.params.get(i).w;
            for (int t = 0; t < 3; t++) {
                int r = pick.nextInt(w.length), c = pick.nextInt(w[0].length);
                double analytic = (w0[i][r][c] - multi.params.get(i).w[r][c]) / lr;
                double numeric = slope(w, r, c, 1e-5, batchLoss);
                // A ReLU can switch on or off inside the finite-difference step (more likely with 8 windows), which distorts the
                // numerical slope but not the true gradient. Disagreement is therefore re-measured with much smaller steps.
                for (double eps = 1e-6; eps >= 1e-8 && disagrees(analytic, numeric, 1e-5); eps /= 10) { numeric = slope(w, r, c, eps, batchLoss); smallerStep++; }
                cmp("batch.param" + i + "[" + r + "," + c + "]", analytic, numeric);
            }
        }
        double diff = maxDifference(single, multi);
        System.out.println("  (" + smallerStep + " smaller-step re-measurement(s) were needed because a ReLU switched inside the step)");
        Adam.PLAIN_SGD_FOR_GRADIENT_CHECK = false;
        System.out.printf("BATCH GRADIENTS (batch of %d, 3 threads): %d weights checked, %d mismatches, worst relative error %.1e %s%n",
            NB, checked - before, bad - badBefore, worst, bad == badBefore ? "(OK)" : "(FAIL)");
        System.out.printf("1 THREAD vs 3 THREADS, one SGD step: batch loss %.12f vs %.12f, largest weight difference %.1e %s%n",
            lossSingle, lossMulti, diff, diff < 1e-12 && Math.abs(lossSingle - lossMulti) < 1e-12 ? "(OK)" : "(FAIL)");

        // the same with Adam for 30 batches, plus evaluation with different thread counts
        MultiLayerTransformer s1 = make(tok, text, T, D, L, H);
        MultiLayerTransformer s3 = make(tok, text, T, D, L, H);
        s3.setThreads(3);
        Random r = new Random(11);
        double lossGap = 0;
        for (int step = 0; step < 30; step++) {
            for (int i = 0; i < NB; i++) { int st = r.nextInt(allIds.length - T - 1); batch[i] = Arrays.copyOfRange(allIds, st, st + T + 1); }
            lossGap = Math.max(lossGap, Math.abs(s1.trainBatch(batch) - s3.trainBatch(batch)));
        }
        double diffAdam = maxDifference(s1, s3);
        String heldOut = text.substring(10000);
        double e1 = s1.evaluate(heldOut, 100);
        s1.setThreads(3);
        double e3 = s1.evaluate(heldOut, 100);
        System.out.printf("30 ADAM BATCHES, 1 thread vs 3 threads: largest loss difference %.1e, largest weight difference %.1e %s%n",
            lossGap, diffAdam, diffAdam < 1e-8 ? "(OK)" : "(FAIL)");
        System.out.printf("EVALUATE, 1 thread vs 3 threads: %.12f vs %.12f %s%n", e1, e3, e1 == e3 ? "(identical, OK)" : "(FAIL)");

        // 4) checkpoints: a saved and reloaded model must score and write exactly like the original
        java.io.File tmp = new java.io.File("gradcheck-tmp.bin");
        try {
            s1.save(tmp.getPath());
            MultiLayerTransformer back = MultiLayerTransformer.load(tmp.getPath());
            double scoreBefore = s1.evaluate(heldOut, 100), scoreAfter = back.evaluate(heldOut, 100);
            String textBefore = s1.generate(text.substring(0, 20), 80, new Random(3), 0.8), textAfter = back.generate(text.substring(0, 20), 80, new Random(3), 0.8);
            boolean same = scoreBefore == scoreAfter && textBefore.equals(textAfter) && back.tokenizer.size() == s1.tokenizer.size();
            System.out.printf("CHECKPOINT, save and reload: evaluation %.12f vs %.12f, same generated text %b %s%n", scoreBefore, scoreAfter, textBefore.equals(textAfter), same ? "(OK)" : "(FAIL)");
        } catch (java.io.IOException e) {
            System.out.println("CHECKPOINT, save and reload: " + e + " (FAIL)");
        } finally {
            tmp.delete();
        }
    }

    static boolean disagrees(double analytic, double numeric, double relativeTolerance) {
        double diff = Math.abs(analytic - numeric);
        return diff > 1e-6 && diff / (Math.abs(analytic) + Math.abs(numeric)) > relativeTolerance;
    }

    static double slope(double[][] w, int r, int c, double eps, DoubleSupplier loss) {
        double orig = w[r][c];
        w[r][c] = orig + eps; double up = loss.getAsDouble();
        w[r][c] = orig - eps; double dn = loss.getAsDouble();
        w[r][c] = orig;
        return (up - dn) / (2 * eps);
    }

    static double[][][] snapshot(MultiLayerTransformer m) {
        double[][][] out = new double[m.params.size()][][];
        for (int i = 0; i < out.length; i++) {
            double[][] w = m.params.get(i).w;
            out[i] = new double[w.length][];
            for (int r = 0; r < w.length; r++) out[i][r] = w[r].clone();
        }
        return out;
    }

    static double maxDifference(MultiLayerTransformer a, MultiLayerTransformer b) {
        double worstDiff = 0;
        for (int i = 0; i < a.params.size(); i++) {
            double[][] x = a.params.get(i).w, y = b.params.get(i).w;
            for (int r = 0; r < x.length; r++) for (int c = 0; c < x[r].length; c++) worstDiff = Math.max(worstDiff, Math.abs(x[r][c] - y[r][c]));
        }
        return worstDiff;
    }
}
