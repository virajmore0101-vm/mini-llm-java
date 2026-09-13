import java.util.*;

/**
 * Self-attention: the mechanism at the heart of GPT/transformers.
 * For each token, it asks "which OTHER tokens in this sentence should
 * I pay attention to?" and blends their information proportionally.
 *
 * Q (query) = "what am I looking for?"
 * K (key)   = "what do I represent?"
 * V (value) = "what do I actually offer if you attend to me?"
 */
public class Attention {

    public static Matrix softmaxRows(Matrix scores) {
        Matrix result = new Matrix(scores.rows(), scores.cols());
        for (int r = 0; r < scores.rows(); r++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < scores.cols(); c++) max = Math.max(max, scores.get(r, c));

            double sum = 0;
            double[] exps = new double[scores.cols()];
            for (int c = 0; c < scores.cols(); c++) {
                exps[c] = Math.exp(scores.get(r, c) - max);
                sum += exps[c];
            }
            for (int c = 0; c < scores.cols(); c++) {
                result.set(r, c, exps[c] / sum);
            }
        }
        return result;
    }

    public static Matrix selfAttention(Matrix embeddings, Matrix Wq, Matrix Wk, Matrix Wv, String[] labels) {
        Matrix Q = embeddings.multiply(Wq);
        Matrix K = embeddings.multiply(Wk);
        Matrix V = embeddings.multiply(Wv);

        Matrix scores = Q.multiply(K.transpose());
        Matrix scaledScores = scores.scale(1.0 / Math.sqrt(Wq.cols()));
        Matrix attentionWeights = softmaxRows(scaledScores);

        System.out.println("\nAttention weights (each row = one token; columns = how much it attends to each token):\n");
        System.out.printf("%-10s", "");
        for (String label : labels) System.out.printf("%8s", label);
        System.out.println();
        for (int r = 0; r < attentionWeights.rows(); r++) {
            System.out.printf("%-10s", labels[r]);
            for (int c = 0; c < attentionWeights.cols(); c++) {
                System.out.printf("%8.2f", attentionWeights.get(r, c));
            }
            System.out.println();
        }

        return attentionWeights.multiply(V);
    }

    public static void main(String[] args) {
        String[] words = {"The", "animal", "crossed", "the", "street", "because", "it", "was", "tired"};

        double[][] embed = {
            {1,   0,   0,   0  }, // The
            {0,   1,   0,   0.2}, // animal
            {0,   0,   1,   0  }, // crossed
            {1,   0,   0,   0  }, // the
            {0,   0,   0,   1  }, // street
            {0.5, 0.5, 0,   0  }, // because
            {0,   0.9, 0,   0.3}, // it   <-- close to "animal"
            {0.3, 0,   0.3, 0  }, // was
            {0,   0,   0.5, 0.5}, // tired
        };
        Matrix embeddings = new Matrix(embed);
        Matrix identity = identityMatrix(4);

        System.out.println("=== Self-attention over: \"" + String.join(" ", words) + "\" ===");
        selfAttention(embeddings, identity, identity, identity, words);

        System.out.println("\nLook at the 'it' row: it should attend most strongly to 'animal' (and itself),");
        System.out.println("because that's the word its embedding was deliberately made similar to.");
    }

    private static Matrix identityMatrix(int n) {
        Matrix m = new Matrix(n, n);
        for (int i = 0; i < n; i++) m.set(i, i, 1.0);
        return m;
    }
}
