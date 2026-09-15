import java.util.*;

public class TrainableAttention {

    public static void main(String[] args) {
        String[] words = {"The", "animal", "crossed", "the", "street", "because", "it", "was", "tired"};
        int d = 4;
        int itIndex = 6;
        int targetIndex = 1;

        Random rand = new Random(42);
        Matrix X = randomMatrix(words.length, d, rand);
        Matrix Wq = randomMatrix(d, d, rand);
        Matrix Wk = randomMatrix(d, d, rand);

        double learningRate = 0.5;
        int epochs = 500;

        for (int epoch = 0; epoch < epochs; epoch++) {
            Matrix Q = X.multiply(Wq);
            Matrix K = X.multiply(Wk);
            Vector qIt = getRow(Q, itIndex);

            double[] scores = new double[words.length];
            for (int j = 0; j < words.length; j++) {
                scores[j] = qIt.dot(getRow(K, j)) / Math.sqrt(d);
            }
            double[] attn = softmax(scores);

            if (epoch % 100 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d: attention on 'animal' = %.3f, loss = %.4f%n",
                    epoch, attn[targetIndex], -Math.log(attn[targetIndex] + 1e-9));
            }

            double[] dScores = new double[words.length];
            for (int j = 0; j < words.length; j++) {
                dScores[j] = attn[j] - (j == targetIndex ? 1.0 : 0.0);
            }

            Vector dQit = new Vector(d);
            for (int j = 0; j < words.length; j++) {
                Vector kj = getRow(K, j);
                for (int p = 0; p < d; p++) {
                    dQit.set(p, dQit.get(p) + dScores[j] * kj.get(p) / Math.sqrt(d));
                }
            }
            Matrix dWq = Matrix.outerProduct(getRow(X, itIndex), dQit);

            Matrix dWk = new Matrix(d, d);
            for (int j = 0; j < words.length; j++) {
                Vector dKj = qIt.scale(dScores[j] / Math.sqrt(d));
                Matrix contribution = Matrix.outerProduct(getRow(X, j), dKj);
                for (int r = 0; r < d; r++) {
                    for (int c = 0; c < d; c++) {
                        dWk.set(r, c, dWk.get(r, c) + contribution.get(r, c));
                    }
                }
            }

            for (int r = 0; r < d; r++) {
                for (int c = 0; c < d; c++) {
                    Wq.set(r, c, Wq.get(r, c) - learningRate * dWq.get(r, c));
                    Wk.set(r, c, Wk.get(r, c) - learningRate * dWk.get(r, c));
                }
            }
        }

        Matrix Q = X.multiply(Wq);
        Matrix K = X.multiply(Wk);
        Vector qIt = getRow(Q, itIndex);
        double[] finalScores = new double[words.length];
        for (int j = 0; j < words.length; j++) {
            finalScores[j] = qIt.dot(getRow(K, j)) / Math.sqrt(d);
        }
        double[] finalAttn = softmax(finalScores);

        System.out.println("\nFinal attention from 'it' after training:");
        for (int j = 0; j < words.length; j++) {
            System.out.printf("  %-10s %.3f%n", words[j], finalAttn[j]);
        }
    }

    private static Vector getRow(Matrix m, int row) {
        Vector v = new Vector(m.cols());
        for (int c = 0; c < m.cols(); c++) v.set(c, m.get(row, c));
        return v;
    }

    private static Matrix randomMatrix(int rows, int cols, Random rand) {
        Matrix m = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m.set(r, c, (rand.nextDouble() * 2 - 1) * 0.5);
        return m;
    }

    private static double[] softmax(double[] scores) {
        double max = Arrays.stream(scores).max().getAsDouble();
        double sum = 0;
        double[] exps = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            exps[i] = Math.exp(scores[i] - max);
            sum += exps[i];
        }
        for (int i = 0; i < scores.length; i++) exps[i] /= sum;
        return exps;
    }
}
