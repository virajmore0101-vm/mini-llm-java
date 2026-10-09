import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Scanner;

public class ChatTransformer {
    // ---- settings you can tweak ----
    static final int TRAIN_CHARS = 200000;      // how much Shakespeare to learn from
    static final int VAL_CHARS = 5000;          // held-out text used only to measure quality
    static final int CONTEXT = 10;              // how many characters the model sees at once
    static final int DIM = 24;
    static final int LAYERS = 3;
    static final int HEADS = 4;
    static final double LEARNING_RATE = 0.004;
    static final int DEFAULT_SECONDS = 420;     // training stops after this long (override: java ChatTransformer 60)
    static final int REPORT_SECONDS = 60;
    static final double TEMPERATURE = 0.8;      // lower = safer and more coherent, higher = wilder
    static final int REPLY_LENGTH = 250;

    public static void main(String[] args) throws IOException {
        int maxSeconds = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_SECONDS;
        String all = Files.readString(Paths.get("shakespeare.txt"));
        String train = all.substring(0, TRAIN_CHARS);
        String val = all.substring(TRAIN_CHARS, TRAIN_CHARS + VAL_CHARS);
        int T = CONTEXT;

        System.out.printf("Training on %,d characters for up to %d seconds (%d layers, %d heads, %d-dim).%n",
            train.length(), maxSeconds, LAYERS, HEADS, DIM);
        MultiLayerTransformer model = new MultiLayerTransformer(train, T, DIM, LAYERS, HEADS, LEARNING_RATE);
        System.out.printf("Held-out loss before training: %.3f (pure guessing would be about %.2f; lower is better)%n",
            model.evaluate(val, 1000), Math.log(model.vocabSize));

        long start = System.currentTimeMillis();
        long nextReport = start + REPORT_SECONDS * 1000L;
        long steps = 0;
        boolean done = false;
        while (!done) {
            for (int i = 0; i + T < train.length(); i++) {
                model.trainStep(train.substring(i, i + T), train.charAt(i + T));
                steps++;
                if ((steps & 255) == 0) {
                    long now = System.currentTimeMillis();
                    if (now - start >= maxSeconds * 1000L) { done = true; break; }
                    if (now >= nextReport) {
                        System.out.printf("  %3ds | %,d steps (%.1f passes over the data) | held-out loss %.3f%n",
                            (now - start) / 1000, steps, (double) steps / (train.length() - T), model.evaluate(val, 1000));
                        nextReport = System.currentTimeMillis() + REPORT_SECONDS * 1000L;
                    }
                }
            }
        }
        System.out.printf("Training stopped after %d seconds and %,d steps. Final held-out loss: %.3f%n",
            (System.currentTimeMillis() - start) / 1000, steps, model.evaluate(val, 1000));
        System.out.println("Ready. Type something (10+ characters works best) and press enter, or 'exit' to quit:\n");

        Random rand = new Random();
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            if (!scanner.hasNextLine()) break;
            String input = scanner.nextLine();
            if (input.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }

            String seed = findSeed(model, train, input, T, rand);
            StringBuilder out = new StringBuilder(seed);
            for (int i = 0; i < REPLY_LENGTH; i++) {
                out.append(model.sampleNext(out.substring(out.length() - T), rand, TEMPERATURE));
            }
            System.out.println("\nModel: " + out + "\n");
        }
        scanner.close();
    }

    private static String findSeed(MultiLayerTransformer model, String text, String input, int T, Random rand) {
        if (input.length() >= T) {
            String candidate = input.substring(input.length() - T);
            boolean known = true;
            for (char c : candidate.toCharArray()) {
                if (!model.charToIndex.containsKey(c)) { known = false; break; }
            }
            if (known) return candidate;
        }
        System.out.println("(Your exact text wasn't all in the training vocabulary -- starting from a real snippet instead.)");
        int randomStart = rand.nextInt(text.length() - T);
        return text.substring(randomStart, randomStart + T);
    }
}
