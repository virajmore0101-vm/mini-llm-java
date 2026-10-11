import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class MultiLayerTransformer {
    static final int EVAL_START = 64;   // validation always predicts characters at index >= 64, so every model sees the same targets
    static final double EMBED_INIT = 0.5, OUT_INIT = 0.1;

    int vocabSize, d, contextLength, numLayers, numHeads;
    Matrix embeddingTable;
    AttentionLayer[] layers;
    FeedForward[] feedforwards;
    LayerNorm[] attnNorms;
    LayerNorm[] ffNorms;
    Matrix Wout;
    Param pEmb, pOut;
    List<Param> params = new ArrayList<>();   // every learnable block, in the same fixed order in every copy of the model
    double learningRate;
    Map<Character, Integer> charToIndex = new HashMap<>();
    char[] indexToChar;

    // Threading: `workers` are copies of this model that share its weights but have their own gradients and their own
    // memory of the last forward pass, so several windows can be processed at the same time without getting in each other's way.
    private int threads = 1;
    private MultiLayerTransformer[] workers;
    private ExecutorService pool;

    /** The vocabulary is every distinct character in `text` (pass the saved vocabulary string when loading a checkpoint). */
    public MultiLayerTransformer(String text, int contextLength, int d, int numLayers, int numHeads, double learningRate) {
        this.contextLength = contextLength;
        this.d = d;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.learningRate = learningRate;

        Set<Character> unique = new TreeSet<>();
        for (char c : text.toCharArray()) unique.add(c);
        vocabSize = unique.size();
        indexToChar = new char[vocabSize];
        int idx = 0;
        for (char c : unique) { charToIndex.put(c, idx); indexToChar[idx] = c; idx++; }

        Random rand = new Random(7);
        embeddingTable = AttentionLayer.randomMatrix(vocabSize, d, rand, EMBED_INIT);
        layers = new AttentionLayer[numLayers];
        feedforwards = new FeedForward[numLayers];
        attnNorms = new LayerNorm[numLayers];
        ffNorms = new LayerNorm[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new AttentionLayer(d, numHeads, rand);
            feedforwards[i] = new FeedForward(d, d * 4, rand);
            attnNorms[i] = new LayerNorm(d);
            ffNorms[i] = new LayerNorm(d);
        }
        Wout = AttentionLayer.randomMatrix(d, vocabSize, rand, OUT_INIT);
        pEmb = new Param(embeddingTable);
        pOut = new Param(Wout);
        collectParams();
    }

    /** A worker copy: same weights (shared, not copied), separate gradients and forward-pass memory. */
    private MultiLayerTransformer(MultiLayerTransformer m) {
        vocabSize = m.vocabSize; d = m.d; contextLength = m.contextLength; numLayers = m.numLayers; numHeads = m.numHeads;
        learningRate = m.learningRate;
        charToIndex = m.charToIndex;    // never changed after construction, so safe to read from every thread
        indexToChar = m.indexToChar;
        embeddingTable = new Matrix(m.embeddingTable.raw());
        Wout = new Matrix(m.Wout.raw());
        layers = new AttentionLayer[numLayers];
        feedforwards = new FeedForward[numLayers];
        attnNorms = new LayerNorm[numLayers];
        ffNorms = new LayerNorm[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new AttentionLayer(m.layers[i]);
            feedforwards[i] = new FeedForward(m.feedforwards[i]);
            attnNorms[i] = new LayerNorm(m.attnNorms[i]);
            ffNorms[i] = new LayerNorm(m.ffNorms[i]);
        }
        pEmb = new Param(embeddingTable);
        pOut = new Param(Wout);
        collectParams();
    }

    private void collectParams() {
        params.add(pEmb);
        params.add(pOut);
        for (int i = 0; i < numLayers; i++) {
            layers[i].collect(params);
            feedforwards[i].collect(params);
            attnNorms[i].collect(params);
            ffNorms[i].collect(params);
        }
    }

    /** Total number of learnable numbers. */
    public int parameterCount() {
        int n = 0;
        for (Param p : params) n += p.size();
        return n;
    }

    /** How many threads training and evaluation may use (1 = everything on the calling thread). */
    public void setThreads(int n) {
        n = Math.max(1, n);
        if (pool != null) { pool.shutdown(); pool = null; }
        workers = null;
        threads = n;
        if (n > 1) {
            workers = new MultiLayerTransformer[n];
            for (int i = 0; i < n; i++) workers[i] = new MultiLayerTransformer(this);
            pool = Executors.newFixedThreadPool(n, r -> { Thread t = new Thread(r); t.setDaemon(true); return t; });
        }
    }

    private interface Job { void run(MultiLayerTransformer worker, int workerId); }

    /** Runs the job once on every worker at the same time and waits for all of them. */
    private void runOnWorkers(Job job) {
        List<Callable<Void>> calls = new ArrayList<>();
        for (int w = 0; w < workers.length; w++) {
            final int id = w;
            calls.add(() -> { job.run(workers[id], id); return null; });
        }
        try {
            for (Future<Void> f : pool.invokeAll(calls)) f.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private Matrix embed(String context) {
        int T = context.length();
        Matrix X = new Matrix(T, d);
        for (int i = 0; i < T; i++) {
            int ci = charToIndex.get(context.charAt(i));
            for (int k = 0; k < d; k++) X.set(i, k, embeddingTable.get(ci, k) + positionalEncoding(i, k));
        }
        return X;
    }

    private double positionalEncoding(int pos, int k) {
        double angle = pos / Math.pow(10000, (2.0 * (k / 2)) / d);
        return (k % 2 == 0) ? Math.sin(angle) : Math.cos(angle);
    }

    private Matrix forwardLayers(Matrix X) {
        Matrix current = X;
        for (int i = 0; i < numLayers; i++) {
            current = layers[i].forward(current);
            current = attnNorms[i].forward(current);
            current = feedforwards[i].forward(current);
            current = ffNorms[i].forward(current);
        }
        return current;
    }

    /** Hidden state of every position (used by GradCheck to prove position i never sees the future). */
    Matrix hiddenStates(String context) { return forwardLayers(embed(context)); }

    /**
     * Forward and backward pass on one window of T+1 characters: the first T are the input, and at EVERY position i
     * the model must predict character i+1. The loss is the average over the T positions.
     * The gradients are ADDED to this model's Params; nothing is updated yet. Returns the window's loss.
     */
    private double accumulate(String window) {
        int n = window.length() - 1;
        Matrix F = forwardLayers(embed(window.substring(0, n)));
        Matrix logits = F.multiply(Wout);

        Matrix dLogits = new Matrix(n, vocabSize);
        double loss = 0;
        for (int i = 0; i < n; i++) {
            double[] probs = softmaxRow(logits, i);
            int target = charToIndex.get(window.charAt(i + 1));
            loss -= Math.log(probs[target] + 1e-9);
            for (int v = 0; v < vocabSize; v++) dLogits.set(i, v, (probs[v] - (v == target ? 1.0 : 0.0)) / n);
        }
        pOut.addGrad(F.transposeMultiply(dLogits));
        Matrix dCurrent = dLogits.multiplyByTranspose(Wout);

        for (int i = numLayers - 1; i >= 0; i--) {
            dCurrent = ffNorms[i].backward(dCurrent);
            dCurrent = feedforwards[i].backward(dCurrent);
            dCurrent = attnNorms[i].backward(dCurrent);
            dCurrent = layers[i].backward(dCurrent);
        }
        for (int i = 0; i < n; i++) {
            double[] row = pEmb.g[charToIndex.get(window.charAt(i))];
            for (int k = 0; k < d; k++) row[k] += dCurrent.get(i, k);
        }
        return loss / n;
    }

    private void zeroGrad() { for (Param p : params) p.zeroGrad(); }

    /** One training step on a single window (gradient, then one Adam update). Returns the window's loss. */
    public double trainSequence(String window) {
        zeroGrad();
        double loss = accumulate(window);
        for (Param p : params) p.step(learningRate);
        return loss;
    }

    /**
     * One training step on a batch of windows: the gradients of all windows are averaged and ONE Adam update is made.
     * With setThreads(n) the windows are processed by n threads at once. Returns the average loss of the batch.
     */
    public double trainBatch(String[] windows) {
        int B = windows.length;
        if (threads <= 1) {
            zeroGrad();
            double loss = 0;
            for (String w : windows) loss += accumulate(w);
            for (Param p : params) { p.scaleGrad(1.0 / B); p.step(learningRate); }
            return loss / B;
        }
        // Phase 1: each worker grabs the next unprocessed window until none are left, adding its gradients to its own Params.
        AtomicInteger next = new AtomicInteger();
        double[] partial = new double[workers.length];
        runOnWorkers((wk, id) -> {
            wk.zeroGrad();
            double s = 0;
            int j;
            while ((j = next.getAndIncrement()) < B) s += wk.accumulate(windows[j]);
            partial[id] = s;
        });
        // Phase 2: each block of weights is combined across workers, averaged and updated by exactly one thread.
        double lr = learningRate;
        runOnWorkers((wk, id) -> {
            for (int i = id; i < params.size(); i += workers.length) {
                Param p = params.get(i);
                p.zeroGrad();
                for (MultiLayerTransformer other : workers) p.addGradFrom(other.params.get(i));
                p.scaleGrad(1.0 / B);
                p.step(lr);
            }
        });
        double loss = 0;
        for (double s : partial) loss += s;
        return loss / B;
    }

    /** Average of sequenceLoss over the windows (what trainBatch's gradient is the derivative of). */
    public double batchLoss(String[] windows) {
        double s = 0;
        for (String w : windows) s += sequenceLoss(w);
        return s / windows.length;
    }

    /** Forward pass only, same average loss as trainSequence (used for gradient checking). */
    public double sequenceLoss(String window) {
        int n = window.length() - 1;
        Matrix logits = forwardLayers(embed(window.substring(0, n))).multiply(Wout);
        double loss = 0;
        for (int i = 0; i < n; i++) loss -= Math.log(softmaxRow(logits, i)[charToIndex.get(window.charAt(i + 1))] + 1e-9);
        return loss / n;
    }

    /** Scores for the next character after `context` (the last contextLength characters are used). */
    private double[] nextLogits(String context) {
        if (context.length() > contextLength) context = context.substring(context.length() - contextLength);
        Matrix F = forwardLayers(embed(context));
        int last = F.rows() - 1;
        double[] logits = new double[vocabSize];
        for (int v = 0; v < vocabSize; v++) {
            double s = 0;
            for (int k = 0; k < d; k++) s += F.get(last, k) * Wout.get(k, v);
            logits[v] = s;
        }
        return logits;
    }

    /**
     * Average loss when predicting the character at positions EVAL_START, EVAL_START+stride, ... of `text`,
     * given the previous contextLength characters. Lower is better; pure guessing is ln(vocabSize).
     * Skips targets whose characters the model has never seen.
     */
    public double evaluate(String text, int maxWindows) {
        int T = contextLength;
        int start = Math.max(EVAL_START, T);
        if (text.length() <= start) return Double.NaN;
        int stride = Math.max(1, (text.length() - start) / maxWindows);
        int count = (text.length() - start + stride - 1) / stride;
        double[] losses = new double[count];
        if (threads <= 1) {
            for (int i = 0; i < count; i++) losses[i] = positionLoss(text, start + i * stride);
        } else {
            runOnWorkers((wk, id) -> {
                for (int i = id; i < count; i += workers.length) losses[i] = wk.positionLoss(text, start + i * stride);
            });
        }
        double sum = 0;
        int n = 0;
        for (double l : losses) if (!Double.isNaN(l)) { sum += l; n++; }   // summed in a fixed order, so the result does not depend on the thread count
        return n == 0 ? Double.NaN : sum / n;
    }

    /** Loss of predicting text[p] from the previous contextLength characters; NaN if any of those characters is unknown. */
    private double positionLoss(String text, int p) {
        int T = contextLength;
        String ctx = text.substring(p - T, p);
        char next = text.charAt(p);
        boolean known = charToIndex.containsKey(next);
        for (int j = 0; j < T && known; j++) known = charToIndex.containsKey(ctx.charAt(j));
        if (!known) return Double.NaN;
        return -Math.log(softmax(nextLogits(ctx))[charToIndex.get(next)] + 1e-9);
    }

    public char predict(String context) {
        double[] logits = nextLogits(context);
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return indexToChar[best];
    }

    /** Temperature below 1.0 plays it safer (more coherent, less varied); above 1.0 gets wilder. */
    public char sampleNext(String context, Random rand, double temperature) {
        double[] logits = nextLogits(context);
        for (int i = 0; i < logits.length; i++) logits[i] /= temperature;
        double[] probs = softmax(logits);
        double r = rand.nextDouble(), cumulative = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) return indexToChar[i];
        }
        return indexToChar[probs.length - 1];
    }

    // ---------- saving and loading (weights only; Adam's memory is not saved) ----------
    public void save(String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeInt(contextLength); out.writeInt(d); out.writeInt(numLayers); out.writeInt(numHeads);
            out.writeUTF(new String(indexToChar));
            writeMatrix(out, embeddingTable); writeMatrix(out, Wout);
            for (int l = 0; l < numLayers; l++) {
                for (int h = 0; h < numHeads; h++) {
                    writeMatrix(out, layers[l].Wq[h]); writeMatrix(out, layers[l].Wk[h]); writeMatrix(out, layers[l].Wv[h]);
                }
                writeMatrix(out, layers[l].Wo); writeMatrix(out, feedforwards[l].W1); writeMatrix(out, feedforwards[l].W2);
                for (LayerNorm ln : new LayerNorm[] { attnNorms[l], ffNorms[l] }) { writeArray(out, ln.gamma); writeArray(out, ln.beta); }
            }
        }
    }

    public static MultiLayerTransformer load(String path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int T = in.readInt(), d = in.readInt(), layers = in.readInt(), heads = in.readInt();
            MultiLayerTransformer m = new MultiLayerTransformer(in.readUTF(), T, d, layers, heads, 0.0);
            readInto(in, m.embeddingTable); readInto(in, m.Wout);
            for (int l = 0; l < layers; l++) {
                for (int h = 0; h < heads; h++) {
                    readInto(in, m.layers[l].Wq[h]); readInto(in, m.layers[l].Wk[h]); readInto(in, m.layers[l].Wv[h]);
                }
                readInto(in, m.layers[l].Wo); readInto(in, m.feedforwards[l].W1); readInto(in, m.feedforwards[l].W2);
                for (LayerNorm ln : new LayerNorm[] { m.attnNorms[l], m.ffNorms[l] }) { readArray(in, ln.gamma); readArray(in, ln.beta); }
            }
            return m;
        }
    }

    private static void writeMatrix(DataOutputStream out, Matrix m) throws IOException {
        out.writeInt(m.rows()); out.writeInt(m.cols());
        for (int r = 0; r < m.rows(); r++) for (int c = 0; c < m.cols(); c++) out.writeDouble(m.get(r, c));
    }
    /** Reads a saved matrix into an existing one (the numbers are shared with Param, so the matrix object must stay the same). */
    private static void readInto(DataInputStream in, Matrix m) throws IOException {
        int r = in.readInt(), c = in.readInt();
        if (r != m.rows() || c != m.cols()) throw new IOException("Saved matrix is " + r + "x" + c + " but the model expects " + m.rows() + "x" + m.cols());
        for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) m.set(i, j, in.readDouble());
    }
    private static void writeArray(DataOutputStream out, double[] a) throws IOException { for (double x : a) out.writeDouble(x); }
    private static void readArray(DataInputStream in, double[] a) throws IOException { for (int i = 0; i < a.length; i++) a[i] = in.readDouble(); }

    private static double[] softmaxRow(Matrix m, int row) {
        double[] a = new double[m.cols()];
        for (int c = 0; c < a.length; c++) a[c] = m.get(row, c);
        return softmax(a);
    }
    private static double[] softmax(double[] scores) {
        double max = Arrays.stream(scores).max().getAsDouble();
        double sum = 0;
        double[] exps = new double[scores.length];
        for (int i = 0; i < scores.length; i++) { exps[i] = Math.exp(scores[i] - max); sum += exps[i]; }
        for (int i = 0; i < scores.length; i++) exps[i] /= sum;
        return exps;
    }

    public static void main(String[] args) {
        String text = "abcabcabcabcabcabcabcabcabcabc";
        int T = 3;
        MultiLayerTransformer model = new MultiLayerTransformer(text, T, 8, 2, 2, 0.01);
        for (int epoch = 0; epoch < 400; epoch++) {
            for (int i = 0; i + T < text.length(); i++) model.trainSequence(text.substring(i, i + T + 1));
        }
        int correct = 0, total = 0;
        for (int i = 0; i + T < text.length(); i++) {
            if (model.predict(text.substring(i, i + T)) == text.charAt(i + T)) correct++;
            total++;
        }
        System.out.println("Sanity check (2 layers, 2 heads each): " + correct + "/" + total);
    }
}
