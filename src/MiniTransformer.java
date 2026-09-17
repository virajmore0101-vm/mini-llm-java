import java.util.*;

public class MiniTransformer {

    int vocabSize, d, contextLength;
    Matrix embeddingTable;
    Matrix Wq, Wk, Wv;
    Matrix Wout;
    double learningRate;
    Map<Character, Integer> charToIndex = new HashMap<>();
    char[] indexToChar;

    public MiniTransformer(String text, int contextLength, int d, double learningRate) {
        this.contextLength = contextLength;
        this.d = d;
        this.learningRate = learningRate;

        Set<Character> unique = new TreeSet<>();
        for (char c : text.toCharArray()) unique.add(c);
        vocabSize = unique.size();
        indexToChar = new char[vocabSize];
        int idx = 0;
        for (char c : unique) {
            charToIndex.put(c, idx);
            indexToChar[idx] = c;
            idx++;
        }

        Random rand = new Random(7);
        embeddingTable = randomMatrix(vocabSize, d, rand, 0.5);
        Wq = randomMatrix(d, d, rand, 0.5);
        Wk = randomMatrix(d, d, rand, 0.5);
        Wv = randomMatrix(d, d, rand, 0.5);
        Wout = randomMatrix(d, vocabSize, rand, 0.5);
    }

    public double trainStep(String context, char actualNext) {
        int T = contextLength;
        Matrix X = new Matrix(T, d);
        for (int i = 0; i < T; i++) {
            int ci = charToIndex.get(context.charAt(i));
            for (int k = 0; k < d; k++) X.set(i, k, embeddingTable.get(ci, k));
        }

        Matrix Q = X.multiply(Wq);
        Matrix K = X.multiply(Wk);
        Matrix V = X.multiply(Wv);

        Vector qLast = getRow(Q, T - 1);
        double[] scores = new double[T];
        for (int j = 0; j < T; j++) scores[j] = qLast.dot(getRow(K, j)) / Math.sqrt(d);
        double[] attn = softmax(scores);

        Vector h = new Vector(d);
        for (int j = 0; j < T; j++) {
            Vector vj = getRow(V, j);
            for (int k = 0; k < d; k++) h.set(k, h.get(k) + attn[j] * vj.get(k));
        }

        Vector logits = Wout.transpose().multiply(h);
        double[] probs = softmax(vectorToArray(logits));
        int targetIdx = charToIndex.get(actualNext);
        double loss = -Math.log(probs[targetIdx] + 1e-9);

        double[] dLogits = probs.clone();
        dLogits[targetIdx] -= 1.0;
        Vector dLogitsVec = arrayToVector(dLogits);

        Matrix dWout = Matrix.outerProduct(h, dLogitsVec);
        Vector dh = Wout.multiply(dLogitsVec);

        double[] dAttn = new double[T];
        for (int j = 0; j < T; j++) dAttn[j] = dh.dot(getRow(V, j));

        Matrix dV = Matrix.outerProduct(arrayToVector(attn), dh);

        double dotSum = 0;
        for (int j = 0; j < T; j++) dotSum += attn[j] * dAttn[j];
        double[] dScores = new double[T];
        for (int j = 0; j < T; j++) dScores[j] = attn[j] * (dAttn[j] - dotSum);

        Vector dQlast = new Vector(d);
        Matrix dK = new Matrix(T, d);
        for (int j = 0; j < T; j++) {
            Vector kj = getRow(K, j);
            for (int p = 0; p < d; p++) dQlast.set(p, dQlast.get(p) + dScores[j] * kj.get(p) / Math.sqrt(d));
            for (int p = 0; p < d; p++) dK.set(j, p, dScores[j] * qLast.get(p) / Math.sqrt(d));
        }

        Matrix dWq = Matrix.outerProduct(getRow(X, T - 1), dQlast);
        Matrix dWk = new Matrix(d, d);
        Matrix dWv = new Matrix(d, d);
        Matrix dX = new Matrix(T, d);
        for (int j = 0; j < T; j++) {
            addInPlace(dWk, Matrix.outerProduct(getRow(X, j), getRow(dK, j)));
            addInPlace(dWv, Matrix.outerProduct(getRow(X, j), getRow(dV, j)));
        }
        Vector dXlastFromQ = Wq.multiply(dQlast);
        for (int j = 0; j < T; j++) {
            Vector dXjFromK = Wk.multiply(getRow(dK, j));
            Vector dXjFromV = Wv.multiply(getRow(dV, j));
            for (int p = 0; p < d; p++) {
                double val = dXjFromK.get(p) + dXjFromV.get(p);
                if (j == T - 1) val += dXlastFromQ.get(p);
                dX.set(j, p, val);
            }
        }

        addScaled(Wout, dWout, -learningRate);
        addScaled(Wq, dWq, -learningRate);
        addScaled(Wk, dWk, -learningRate);
        addScaled(Wv, dWv, -learningRate);
        for (int i = 0; i < T; i++) {
            int ci = charToIndex.get(context.charAt(i));
            for (int k = 0; k < d; k++) {
                embeddingTable.set(ci, k, embeddingTable.get(ci, k) - learningRate * dX.get(i, k));
            }
        }

        return loss;
    }

    public char predict(String context) {
        int T = contextLength;
        Matrix X = new Matrix(T, d);
        for (int i = 0; i < T; i++) {
            int ci = charToIndex.get(context.charAt(i));
            for (int k = 0; k < d; k++) X.set(i, k, embeddingTable.get(ci, k));
        }
        Matrix Q = X.multiply(Wq);
        Matrix K = X.multiply(Wk);
        Matrix V = X.multiply(Wv);
        Vector qLast = getRow(Q, T - 1);
        double[] scores = new double[T];
        for (int j = 0; j < T; j++) scores[j] = qLast.dot(getRow(K, j)) / Math.sqrt(d);
        double[] attn = softmax(scores);
        Vector h = new Vector(d);
        for (int j = 0; j < T; j++) {
            Vector vj = getRow(V, j);
            for (int k = 0; k < d; k++) h.set(k, h.get(k) + attn[j] * vj.get(k));
        }
        Vector logits = Wout.transpose().multiply(h);
        double[] probs = softmax(vectorToArray(logits));
        int best = 0;
        for (int i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
        return indexToChar[best];
    }

    public char sampleNext(String context, Random rand) {
        int T = contextLength;
        Matrix X = new Matrix(T, d);
        for (int i = 0; i < T; i++) {
            int ci = charToIndex.get(context.charAt(i));
            for (int k = 0; k < d; k++) X.set(i, k, embeddingTable.get(ci, k));
        }
        Matrix Q = X.multiply(Wq);
        Matrix K = X.multiply(Wk);
        Matrix V = X.multiply(Wv);
        Vector qLast = getRow(Q, T - 1);
        double[] scores = new double[T];
        for (int j = 0; j < T; j++) scores[j] = qLast.dot(getRow(K, j)) / Math.sqrt(d);
        double[] attn = softmax(scores);
        Vector h = new Vector(d);
        for (int j = 0; j < T; j++) {
            Vector vj = getRow(V, j);
            for (int k = 0; k < d; k++) h.set(k, h.get(k) + attn[j] * vj.get(k));
        }
        Vector logits = Wout.transpose().multiply(h);
        double[] probs = softmax(vectorToArray(logits));

        double r = rand.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) return indexToChar[i];
        }
        return indexToChar[probs.length - 1];
    }

    private static void addInPlace(Matrix a, Matrix b) {
        for (int r = 0; r < a.rows(); r++) for (int c = 0; c < a.cols(); c++) a.set(r, c, a.get(r, c) + b.get(r, c));
    }
    private static void addScaled(Matrix a, Matrix b, double scale) {
        for (int r = 0; r < a.rows(); r++) for (int c = 0; c < a.cols(); c++) a.set(r, c, a.get(r, c) + scale * b.get(r, c));
    }
    private static Vector getRow(Matrix m, int row) {
        Vector v = new Vector(m.cols());
        for (int c = 0; c < m.cols(); c++) v.set(c, m.get(row, c));
        return v;
    }
    private static double[] vectorToArray(Vector v) {
        double[] a = new double[v.size()];
        for (int i = 0; i < v.size(); i++) a[i] = v.get(i);
        return a;
    }
    private static Vector arrayToVector(double[] a) {
        Vector v = new Vector(a.length);
        for (int i = 0; i < a.length; i++) v.set(i, a[i]);
        return v;
    }
    private static Matrix randomMatrix(int rows, int cols, Random rand, double scale) {
        Matrix m = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) m.set(r, c, (rand.nextDouble() * 2 - 1) * scale);
        return m;
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
        MiniTransformer model = new MiniTransformer(text, T, 8, 0.1);
        int epochs = 300;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i + T < text.length(); i++) {
                model.trainStep(text.substring(i, i + T), text.charAt(i + T));
            }
        }
        int correct = 0, total = 0;
        for (int i = 0; i + T < text.length(); i++) {
            if (model.predict(text.substring(i, i + T)) == text.charAt(i + T)) correct++;
            total++;
        }
        System.out.println("Sanity check: " + correct + "/" + total);
    }
}
