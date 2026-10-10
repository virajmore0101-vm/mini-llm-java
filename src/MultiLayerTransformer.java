import java.io.*;
import java.util.*;

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
    Adam adamOut, adamEmb;
    double learningRate;
    Map<Character, Integer> charToIndex = new HashMap<>();
    char[] indexToChar;

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
        adamOut = new Adam(d * vocabSize);
        adamEmb = new Adam(vocabSize * d);
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
     * One training step on a window of T+1 characters: the first T are the input, and at EVERY position i
     * the model must predict character i+1. The loss is the average over the T positions.
     */
    public double trainSequence(String window) {
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
        Matrix dWout = F.transposeMultiply(dLogits);
        Matrix dCurrent = dLogits.multiplyByTranspose(Wout);

        for (int i = numLayers - 1; i >= 0; i--) {
            dCurrent = ffNorms[i].backward(dCurrent, learningRate);
            dCurrent = feedforwards[i].backward(dCurrent, learningRate);
            dCurrent = attnNorms[i].backward(dCurrent, learningRate);
            dCurrent = layers[i].backward(dCurrent, learningRate);
        }

        Wout = adamOut.step(Wout, dWout, learningRate);
        Matrix dEmbedding = new Matrix(vocabSize, d);
        for (int i = 0; i < n; i++) {
            int ci = charToIndex.get(window.charAt(i));
            for (int k = 0; k < d; k++) dEmbedding.set(ci, k, dEmbedding.get(ci, k) + dCurrent.get(i, k));
        }
        embeddingTable = adamEmb.step(embeddingTable, dEmbedding, learningRate);
        return loss / n;
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
        int stride = Math.max(1, (text.length() - start) / maxWindows);
        double sum = 0;
        int n = 0;
        for (int p = start; p < text.length(); p += stride) {
            String ctx = text.substring(p - T, p);
            char next = text.charAt(p);
            boolean known = charToIndex.containsKey(next);
            for (int j = 0; j < T && known; j++) known = charToIndex.containsKey(ctx.charAt(j));
            if (!known) continue;
            sum -= Math.log(softmax(nextLogits(ctx))[charToIndex.get(next)] + 1e-9);
            n++;
        }
        return n == 0 ? Double.NaN : sum / n;
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
            m.embeddingTable = readMatrix(in); m.Wout = readMatrix(in);
            for (int l = 0; l < layers; l++) {
                for (int h = 0; h < heads; h++) {
                    m.layers[l].Wq[h] = readMatrix(in); m.layers[l].Wk[h] = readMatrix(in); m.layers[l].Wv[h] = readMatrix(in);
                }
                m.layers[l].Wo = readMatrix(in); m.feedforwards[l].W1 = readMatrix(in); m.feedforwards[l].W2 = readMatrix(in);
                for (LayerNorm ln : new LayerNorm[] { m.attnNorms[l], m.ffNorms[l] }) { readArray(in, ln.gamma); readArray(in, ln.beta); }
            }
            return m;
        }
    }

    private static void writeMatrix(DataOutputStream out, Matrix m) throws IOException {
        out.writeInt(m.rows()); out.writeInt(m.cols());
        for (int r = 0; r < m.rows(); r++) for (int c = 0; c < m.cols(); c++) out.writeDouble(m.get(r, c));
    }
    private static Matrix readMatrix(DataInputStream in) throws IOException {
        Matrix m = new Matrix(in.readInt(), in.readInt());
        for (int r = 0; r < m.rows(); r++) for (int c = 0; c < m.cols(); c++) m.set(r, c, in.readDouble());
        return m;
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
