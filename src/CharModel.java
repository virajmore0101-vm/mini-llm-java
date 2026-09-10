import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * A character-level language model built from simple counting, not
 * gradient descent. Given the last N characters, it tracks which
 * character came next in the training text, then generates new text
 * by repeatedly sampling from those learned distributions.
 *
 * This is the same fundamental idea as GPT's next-token prediction --
 * just using frequency counts instead of a trained neural network.
 */
public class CharModel {

    private final int contextLength;
    private final Map<String, Map<Character, Integer>> transitions = new HashMap<>();
    private final Random random = new Random();

    public CharModel(int contextLength) {
        this.contextLength = contextLength;
    }

    /** Slide a window over the text, counting what character follows each context. */
    public void train(String text) {
        for (int i = 0; i + contextLength < text.length(); i++) {
            String context = text.substring(i, i + contextLength);
            char next = text.charAt(i + contextLength);
            transitions.computeIfAbsent(context, k -> new HashMap<>())
                       .merge(next, 1, Integer::sum);
        }
    }

    /** Generate new text by repeatedly sampling the next character. */
    public String generate(String seed, int length) {
        StringBuilder output = new StringBuilder(seed);
        String context = seed;
        for (int i = 0; i < length; i++) {
            Map<Character, Integer> nextCounts = transitions.get(context);
            if (nextCounts == null) break;
            char next = sample(nextCounts);
            output.append(next);
            context = output.substring(output.length() - contextLength);
        }
        return output.toString();
    }

    /** Weighted random pick: characters that followed this context more often get picked more often. */
    private char sample(Map<Character, Integer> counts) {
        int total = counts.values().stream().mapToInt(Integer::intValue).sum();
        int r = random.nextInt(total);
        int cumulative = 0;
        for (Map.Entry<Character, Integer> entry : counts.entrySet()) {
            cumulative += entry.getValue();
            if (r < cumulative) return entry.getKey();
        }
        return counts.keySet().iterator().next();
    }

    public static void main(String[] args) throws IOException {
        String text = Files.readString(Paths.get("shakespeare.txt"));
        System.out.println("Loaded " + text.length() + " characters of training text.");

        int contextLength = 6; // try 3 for more chaos, 10 for more memorization
        CharModel model = new CharModel(contextLength);

        long start = System.currentTimeMillis();
        model.train(text);
        System.out.println("Learned patterns in " + (System.currentTimeMillis() - start) + " ms.");

        String seed = text.substring(0, contextLength);
        String generated = model.generate(seed, 500);

        System.out.println("\n=== Generated text ===\n");
        System.out.println(generated);
    }
}
