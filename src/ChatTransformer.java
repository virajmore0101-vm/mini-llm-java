import java.io.IOException;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.*;

public class ChatTransformer {
    // ---- settings you can tweak ----
    static final int TRAIN_CHARS = 200000;      // characters of Shakespeare to learn from
    static final int VAL_CHARS = 5000;          // held-out text used only to measure quality
    static final int DEFAULT_CONTEXT = 32;      // how many characters the model sees at once
    static final int DIM = 24;
    static final int LAYERS = 3;
    static final int HEADS = 4;
    static final double LEARNING_RATE = 0.001;  // Adam, at the start of training
    static final double LR_END_FRACTION = 0.1;  // the learning rate shrinks linearly to this fraction by the end, which calms the final-loss wobble
    static final int DEFAULT_SECONDS = 420;
    static final int REPORT_SECONDS = 60;
    static final double TEMPERATURE = 0.8;
    static final int REPLY_LENGTH = 250;
    static final long SEED = 123;               // fixed seed: same settings give the same run
    static final String BEST = "best.bin", LAST = "last.bin", LOG = "results.log";
    static final Pattern WORD = Pattern.compile("[a-z]+(?:'[a-z]+)*");

    // usage:  java ChatTransformer [seconds] [context] [learningRate]     train, then chat
    //         java ChatTransformer chat                                   chat with the saved best.bin (no training)
    public static void main(String[] args) throws IOException {
        if (args.length > 0 && args[0].equals("chat")) {
            chat(MultiLayerTransformer.load(BEST), "saved best.bin");
            return;
        }
        int maxSeconds = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_SECONDS;
        int T = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_CONTEXT;
        double lr = args.length > 2 ? Double.parseDouble(args[2]) : LEARNING_RATE;

        String all = Files.readString(Paths.get("shakespeare.txt"));
        String train = all.substring(0, TRAIN_CHARS);
        String val = all.substring(TRAIN_CHARS, TRAIN_CHARS + VAL_CHARS);
        String probe = train.substring(100000, 100000 + VAL_CHARS);   // training text, measured the same way as val
        Set<String> realWords = new HashSet<>();
        Matcher wm = WORD.matcher(all.toLowerCase());
        while (wm.find()) realWords.add(wm.group());

        String settings = String.format("context=%d dim=%d layers=%d heads=%d lr=%s", T, DIM, LAYERS, HEADS, lr);
        System.out.printf("Training %,d characters for up to %d seconds (%s).%n", train.length(), maxSeconds, settings);
        MultiLayerTransformer model = new MultiLayerTransformer(train, T, DIM, LAYERS, HEADS, lr);
        System.out.printf("Before training: val loss %.3f (pure guessing = %.2f; lower is better)%n",
            model.evaluate(val, 1000), Math.log(model.vocabSize));
        System.out.println("  passes = how many times the text's characters have been used as prediction targets");

        Random rng = new Random(SEED);
        long start = System.currentTimeMillis(), nextReport = start + REPORT_SECONDS * 1000L;
        long steps = 0, bestStep = 0;
        double runLoss = 0; int runN = 0, bestSeconds = 0;
        double bestVal = Double.MAX_VALUE;
        double[] st = {0, 0};
        boolean done = false;
        while (!done) {
            int s = rng.nextInt(train.length() - T - 1);                   // random window, so no part of the text is favored
            runLoss += model.trainSequence(train.substring(s, s + T + 1));
            runN++; steps++;
            if ((steps & 127) != 0) continue;
            long now = System.currentTimeMillis();
            done = now - start >= maxSeconds * 1000L;
            model.learningRate = lr * (1 - (1 - LR_END_FRACTION) * Math.min(1.0, (now - start) / (maxSeconds * 1000.0)));
            if (now >= nextReport || done) {
                double v = model.evaluate(val, 1000);
                st = textStats(model, val.substring(0, T), realWords, train);
                System.out.printf("%4ds | %,9d steps | %5.1f passes | train %.3f | train-eval %.3f | val %.3f | real words %2.0f%% | copied %2.0f%%%n",
                    (now - start) / 1000, steps, (double) steps * T / train.length(), runLoss / runN,
                    model.evaluate(probe, 1000), v, st[0], st[1]);
                if (v < bestVal) { bestVal = v; bestStep = steps; bestSeconds = (int) ((now - start) / 1000); model.save(BEST); }
                runLoss = 0; runN = 0;
                nextReport = System.currentTimeMillis() + REPORT_SECONDS * 1000L;
            }
        }
        model.save(LAST);
        double finalVal = model.evaluate(val, 1000), finalTrain = model.evaluate(probe, 1000);
        System.out.printf("Done. %,d steps. Final val %.3f (train-eval %.3f). Best val %.3f at %ds, saved to %s.%n",
            steps, finalVal, finalTrain, bestVal, bestSeconds, BEST);
        String sample = generate(model, val.substring(0, T), 300, new Random(99));
        System.out.println("Sample (temperature " + TEMPERATURE + "):\n" + sample + "\n");

        String line = String.format("%s | %s | %ds %,d steps | best val %.3f @%ds | final val %.3f | train-eval %.3f | real words %.0f%% | copied %.0f%%%n",
            LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")), settings, maxSeconds, steps,
            bestVal, bestSeconds, finalVal, finalTrain, st[0], st[1]);
        Files.writeString(Paths.get(LOG), line, StandardOpenOption.CREATE, StandardOpenOption.APPEND);

        chat(MultiLayerTransformer.load(BEST), String.format("best checkpoint (val %.3f)", bestVal));
    }

    static String generate(MultiLayerTransformer model, String seed, int length, Random rand) {
        StringBuilder out = new StringBuilder(seed);
        for (int i = 0; i < length; i++) {
            out.append(model.sampleNext(out.substring(Math.max(0, out.length() - model.contextLength)), rand, TEMPERATURE));
        }
        return out.toString();
    }

    /** {percent of generated words that appear somewhere in Shakespeare, percent of generated 20-character pieces copied verbatim from the training text} */
    static double[] textStats(MultiLayerTransformer model, String seed, Set<String> realWords, String train) {
        String text = generate(model, seed, 1000, new Random(99)).substring(seed.length());
        List<String> words = new ArrayList<>();
        Matcher m = WORD.matcher(text.toLowerCase());
        while (m.find()) words.add(m.group());
        if (words.size() > 2) words = words.subList(1, words.size() - 1);   // first and last may be cut in half
        int hits = 0;
        for (String w : words) if (realWords.contains(w)) hits++;
        int copied = 0, pieces = text.length() - 19;
        for (int i = 0; i < pieces; i++) if (train.contains(text.substring(i, i + 20))) copied++;
        return new double[] { words.isEmpty() ? 0 : 100.0 * hits / words.size(), 100.0 * copied / pieces };
    }

    static void chat(MultiLayerTransformer model, String which) {
        System.out.println("Chatting with the " + which + ". Note: this model CONTINUES text in Shakespeare's style; it cannot answer questions yet.");
        System.out.println("Type something and press enter, or 'exit' to quit:\n");
        Random rand = new Random();
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            if (!scanner.hasNextLine()) break;
            String input = scanner.nextLine();
            if (input.equalsIgnoreCase("exit")) { System.out.println("Goodbye!"); break; }

            StringBuilder out = new StringBuilder();
            int skipped = 0;
            for (char c : input.toCharArray()) { if (model.charToIndex.containsKey(c)) out.append(c); else skipped++; }
            if (skipped > 0) System.out.println("(Skipped " + skipped + " character(s) the model has never seen.)");
            if (out.length() == 0) out.append('\n');
            for (int i = 0; i < REPLY_LENGTH; i++) {
                out.append(model.sampleNext(out.substring(Math.max(0, out.length() - model.contextLength)), rand, TEMPERATURE));
            }
            System.out.println("\nModel: " + out + "\n");
        }
        scanner.close();
    }
}
