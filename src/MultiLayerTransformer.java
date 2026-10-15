import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class MultiLayerTransformer {
    static final int EVAL_START = 64;   // validation only scores text from character 64 on, so every model is scored on the same part of the passage
    static final double EMBED_INIT = 0.5, OUT_INIT = 0.1;
    static final int FILE_FORMAT = 2;   // checkpoint files start with -FILE_FORMAT; older files start directly with the context length

    int vocabSize, d, contextLength, numLayers, numHeads;
    Tokenizer tokenizer;
    Matrix embeddingTable;
    AttentionLayer[] layers;
    FeedForward[] feedforwards;
    LayerNorm[] attnNorms;
    LayerNorm[] ffNorms;
    Matrix Wout;
    Param pEmb, pOut;
    List<Param> params = new ArrayList<>();   // every learnable block, in the same fixed order in every copy of the model
    double learningRate;

    // Threading: `workers` are copies of this model that share its weights but have their own gradients and their own
    // memory of the last forward pass, so several windows can be processed at the same time without getting in each other's way.
    private int threads = 1;
    private MultiLayerTransformer[] workers;
    private ExecutorService pool;

    /** Character-level model: the vocabulary is every distinct character in `text`. */
    public MultiLayerTransformer(String text, int contextLength, int d, int numLayers, int numHeads, double learningRate) {
        this(Tokenizer.chars(text), contextLength, d, numLayers, numHeads, learningRate);
    }

    /** Model over any tokenizer (single characters, or word-pieces learned with Tokenizer.train). */
    public MultiLayerTransformer(Tokenizer tokenizer, int contextLength, int d, int numLayers, int numHeads, double learningRate) {
        this.tokenizer = tokenizer;
        this.vocabSize = tokenizer.size();
        this.contextLength = contextLength;
        this.d = d;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.learningRate = learningRate;

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
        tokenizer = m.tokenizer;        // never changed after construction, so safe to use from every thread
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

    /** Embeddings (plus position encodings) of the first n token ids. */
    private Matrix embed(int[] ids, int n) {
        Matrix X = new Matrix(n, d);
        for (int i = 0; i < n; i++) {
            int ci = ids[i];
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
    Matrix hiddenStatesIds(int[] ids) { return forwardLayers(embed(ids, ids.length)); }
    Matrix hiddenStates(String context) { return hiddenStatesIds(tokenizer.encode(context)); }

    /**
     * Forward and backward pass on one window of T+1 tokens: the first T are the input, and at EVERY position i
     * the model must predict token i+1. The loss is the average over the T positions.
     * The gradients are ADDED to this model's Params; nothing is updated yet. Returns the window's loss.
     */
    private double accumulate(int[] window) {
        int n = window.length - 1;
        Matrix F = forwardLayers(embed(window, n));
        Matrix logits = F.multiply(Wout);

        Matrix dLogits = new Matrix(n, vocabSize);
        double loss = 0;
        for (int i = 0; i < n; i++) {
            double[] probs = softmaxRow(logits, i);
            int target = window[i + 1];
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
            double[] row = pEmb.g[window[i]];
            for (int k = 0; k < d; k++) row[k] += dCurrent.get(i, k);
        }
        return loss / n;
    }

    private void zeroGrad() { for (Param p : params) p.zeroGrad(); }

    /** One training step on a single window (gradient, then one Adam update). Returns the window's loss (per token). */
    public double trainSequence(int[] window) {
        zeroGrad();
        double loss = accumulate(window);
        for (Param p : params) p.step(learningRate);
        return loss;
    }

    public double trainSequence(String window) { return trainSequence(tokenizer.encode(window)); }

    /**
     * One training step on a batch of windows: the gradients of all windows are averaged and ONE Adam update is made.
     * With setThreads(n) the windows are processed by n threads at once. Returns the average loss (per token) of the batch.
     */
    public double trainBatch(int[][] windows) {
        int B = windows.length;
        if (threads <= 1) {
            zeroGrad();
            double loss = 0;
            for (int[] w : windows) loss += accumulate(w);
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

    public double trainBatch(String[] windows) {
        int[][] ids = new int[windows.length][];
        for (int i = 0; i < ids.length; i++) ids[i] = tokenizer.encode(windows[i]);
        return trainBatch(ids);
    }

    /** Forward pass only, same average loss as trainSequence (used for gradient checking). */
    public double sequenceLoss(int[] window) {
        int n = window.length - 1;
        Matrix logits = forwardLayers(embed(window, n)).multiply(Wout);
        double loss = 0;
        for (int i = 0; i < n; i++) loss -= Math.log(softmaxRow(logits, i)[window[i + 1]] + 1e-9);
        return loss / n;
    }

    public double sequenceLoss(String window) { return sequenceLoss(tokenizer.encode(window)); }

    /** Average of sequenceLoss over the windows (what trainBatch's gradient is the derivative of). */
    public double batchLoss(int[][] windows) {
        double s = 0;
        for (int[] w : windows) s += sequenceLoss(w);
        return s / windows.length;
    }

    /** Scores for the next token after `context` (the last contextLength tokens are used). */
    private double[] nextLogits(int[] context) {
        int[] c = context.length > contextLength ? Arrays.copyOfRange(context, context.length - contextLength, context.length) : context;
        Matrix F = forwardLayers(embed(c, c.length));
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
     * Average loss PER CHARACTER of the model on `text`, in nats (lower is better; guessing at random is ln(vocabulary) per token).
     * Tokens at positions EVAL_START, ... of the text (and with a full context of contextLength tokens before them) are predicted
     * and scored; the total loss is divided by the number of characters those tokens stand for, so a character-level model and a
     * word-piece model can be compared on the same passage. Characters without a token are skipped.
     */
    public double evaluate(String text, int maxWindows) {
        int T = contextLength;
        int[] ids = tokenizer.encode(text);
        int[] charStart = new int[ids.length + 1];
        for (int i = 0; i < ids.length; i++) charStart[i + 1] = charStart[i] + tokenizer.tokenText(ids[i]).length();
        int start = T;
        while (start < ids.length && charStart[start] < EVAL_START) start++;
        if (ids.length <= start) return Double.NaN;
        int stride = Math.max(1, (ids.length - start) / maxWindows);
        int count = (ids.length - start + stride - 1) / stride;
        double[] losses = new double[count];
        final int firstPosition = start, step = stride;
        if (threads <= 1) {
            for (int i = 0; i < count; i++) losses[i] = positionLoss(ids, firstPosition + i * step);
        } else {
            runOnWorkers((wk, id) -> {
                for (int i = id; i < count; i += workers.length) losses[i] = wk.positionLoss(ids, firstPosition + i * step);
            });
        }
        double sum = 0;
        long chars = 0;                                   // summed in a fixed order, so the result does not depend on the thread count
        for (int i = 0; i < count; i++) { sum += losses[i]; chars += tokenizer.tokenText(ids[firstPosition + i * step]).length(); }
        return sum / chars;
    }

    /** Loss of predicting ids[p] from the previous contextLength tokens. */
    private double positionLoss(int[] ids, int p) {
        int[] ctx = Arrays.copyOfRange(ids, p - contextLength, p);
        return -Math.log(softmax(nextLogits(ctx))[ids[p]] + 1e-9);
    }

    /** The most likely next character after `context` (the first character of the most likely next token). */
    public char predict(String context) {
        double[] logits = nextLogits(tokenizer.encode(context));
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return tokenizer.tokenText(best).charAt(0);
    }

    /** Temperature below 1.0 plays it safer (more coherent, less varied); above 1.0 gets wilder. */
    public int sampleNextId(int[] context, Random rand, double temperature) {
        double[] logits = nextLogits(context);
        for (int i = 0; i < logits.length; i++) logits[i] /= temperature;
        double[] probs = softmax(logits);
        double r = rand.nextDouble(), cumulative = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) return i;
        }
        return probs.length - 1;
    }

    /**
     * Continues `prompt` for `numChars` characters and returns only the continuation. Characters of the prompt that have no
     * token are skipped; the prompt must contain at least one known character.
     */
    public String generate(String prompt, int numChars, Random rand, double temperature) {
        int[] start = tokenizer.encode(prompt);
        if (start.length == 0) throw new IllegalArgumentException("The prompt has no characters the model knows.");
        int[] buf = Arrays.copyOf(start, start.length + numChars);   // a token is at least one character, so numChars tokens is enough
        int len = start.length;
        StringBuilder out = new StringBuilder();
        while (out.length() < numChars) {
            int from = Math.max(0, len - contextLength);
            int id = sampleNextId(Arrays.copyOfRange(buf, from, len), rand, temperature);
            buf[len++] = id;
            out.append(tokenizer.tokenText(id));
        }
        return out.substring(0, numChars);
    }

    // ---------- saving and loading (weights only; Adam's memory is not saved) ----------
    public void save(String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeInt(-FILE_FORMAT);
            out.writeInt(contextLength); out.writeInt(d); out.writeInt(numLayers); out.writeInt(numHeads);
            tokenizer.write(out);
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

    /** Loads files written by this version and by the older character-level versions. */
    public static MultiLayerTransformer load(String path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int first = in.readInt();
            int T = first < 0 ? in.readInt() : first;           // old files start directly with the context length
            int d = in.readInt(), layers = in.readInt(), heads = in.readInt();
            Tokenizer tok = first < 0 ? Tokenizer.read(in) : Tokenizer.chars(in.readUTF());
            MultiLayerTransformer m = new MultiLayerTransformer(tok, T, d, layers, heads, 0.0);
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
