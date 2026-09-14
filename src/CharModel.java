import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class CharModel {

    private final int contextLength;
    private final Map<String, Map<Character, Integer>> transitions = new HashMap<>();
    private final Random random = new Random();

    public CharModel(int contextLength) {
        this.contextLength = contextLength;
    }

    public void train(String text) {
        for (int i = 0; i + contextLength < text.length(); i++) {
            String context = text.substring(i, i + contextLength);
            char next = text.charAt(i + contextLength);
            transitions.computeIfAbsent(context, k -> new HashMap<>())
                       .merge(next, 1, Integer::sum);
        }
    }

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

    /** Whether this exact context was seen during training (used by the chat loop to fall back gracefully). */
    public boolean knowsContext(String context) {
        return transitions.containsKey(context);
    }

    public static void main(String[] args) throws IOException {
        String text = Files.readString(Paths.get("shakespeare.txt"));
        System.out.println("Loaded " + text.length() + " characters of training text.");

        int contextLength = 6;
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
