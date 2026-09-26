import java.util.*;

public class MultiLayerTransformer {
    int vocabSize, d, contextLength, numLayers;
    Matrix embeddingTable;
    AttentionLayer[] layers;
    Matrix Wout;
    double learningRate;
    Map<Character, Integer> charToIndex = new HashMap<>();
    char[] indexToChar;

    public MultiLayerTransformer(String text, int contextLength, int d, int numLayers, double learningRate) {
        this.contextLength = contextLength;
        this.d = d;
        this.numLayers = numLayers;
        this.learningRate = learningRate;

        Set<Character> unique = new TreeSet<>();
        for (char c : text.toCharArray()) unique.add(c);
        vocabSize = unique.size();
        indexToChar = new char[vocabSize];
        int idx = 0;
        for (char c : unique) { charToIndex.put(c, idx); indexToChar[idx] = c; idx++; }

        Random rand = new Random(7);
        embeddingTable = AttentionLayer.randomMatrix(vocabSize, d, rand, 0.5);
        layers = new AttentionLayer[numLayers];
        for (int i = 0; i < numLayers; i++) layers[i] = new AttentionLayer(d, rand);
        Wout = AttentionLayer.randomMatrix(d, vocabSize, rand, 0.5);
    }

    private Matrix embed(String context) {
        int T = contextLength;
        Matrix X = new Matrix(T, d);
        for (int i = 0; i < T; i++) {
            int ci = charToIndex.get(context.charAt(i));
            for (int k = 0; k < d; k++) X.set(i, k, embeddingTable.get(ci, k));
        }
        return X;
    }

    private Matrix forwardLayers(Matrix X) {
        Matrix current = X;
        for (int i = 0; i < numLayers; i++) current = layers[i].forward(current);
        return current;
    }

    public double trainStep(String context, char actualNext) {
        int T = contextLength;
        Matrix X = embed(context);
        Matrix finalOutput = forwardLayers(X);

        Vector h = getRow(finalOutput, T - 1);
        Vector logits = Wout.transpose().multiply(h);
        double[] probs = softmax(vectorToArray(logits));
        int targetIdx = charToIndex.get(actualNext);
        double loss = -Math.log(probs[targetIdx] + 1e-9);

        double[] dLogits = probs.clone();
        dLogits[targetIdx] -= 1.0;
        Vector dLogitsVec = arrayToVector(dLogits);
        Matrix dWout = Matrix.outerProduct(h, dLogitsVec);
        Vector dh = Wout.multiply(dLogitsVec);

        Matrix dCurrent = new Matrix(T, d);
        for (int k = 0; k < d; k++) dCurrent.set(T - 1, k, dh.get(k));

        for (int i = numLayers - 1; i >= 0; i--) {
            dCurrent = layers[i].backward(dCurrent, learningRate);
        }

        Wout = Wout.add(dWout.scale(-learningRate));
        for (int i = 0; i < T; i++) {
            int ci = charToIndex.get(context.charAt(i));
            for (int k = 0; k < d; k++) {
                embeddingTable.set(ci, k, embeddingTable.get(ci, k) - learningRate * dCurrent.get(i, k));
            }
        }
        return loss;
    }

    public char predict(String context) {
        Matrix finalOutput = forwardLayers(embed(context));
        Vector h = getRow(finalOutput, contextLength - 1);
        double[] probs = softmax(vectorToArray(Wout.transpose().multiply(h)));
        int best = 0;
        for (int i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
        return indexToChar[best];
    }

    public char sampleNext(String context, Random rand) {
        Matrix finalOutput = forwardLayers(embed(context));
        Vector h = getRow(finalOutput, contextLength - 1);
        double[] probs = softmax(vectorToArray(Wout.transpose().multiply(h)));
        double r = rand.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) return indexToChar[i];
        }
        return indexToChar[probs.length - 1];
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
        MultiLayerTransformer model = new MultiLayerTransformer(text, T, 8, 2, 0.05);

        int epochs = 400;
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
        System.out.println("Sanity check (2 stacked layers): " + correct + "/" + total);
    }
}
