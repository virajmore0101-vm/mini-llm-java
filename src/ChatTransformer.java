import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.*;

public class ChatTransformer {
    // ---- defaults (override on the command line, e.g.  java ChatTransformer data=tinystories.txt chars=2000000 seconds=420) ----
    static final String DEFAULT_DATA = "shakespeare.txt";
    static final int DEFAULT_CHARS = 200000;    // how much of the file to train on
    static final int DEFAULT_VAL_CHARS = 5000;  // the next piece of the file, held out and used only to measure quality
    static final int DEFAULT_CONTEXT = 32;      // how many characters the model sees at once
    static final int DIM = 24;
    static final int LAYERS = 3;
    static final int HEADS = 4;
    static final double LEARNING_RATE = 0.001;  // Adam, at the start of training
    static final double LR_END_FRACTION = 0.1;  // the learning rate shrinks linearly to this fraction by the end
    static final int DEFAULT_SECONDS = 420;
    static final int REPORT_SECONDS = 60;
    static final int MIN_CHAR_COUNT = 3;        // characters seen fewer times than this are removed from the data
    static final double TEMPERATURE = 0.8;
    static final int REPLY_LENGTH = 250;
    static final long SEED = 123;               // fixed seed: same settings give the same run
    static final String LOG = "results.log";
    static final Pattern WORD = Pattern.compile("[a-z]+(?:'[a-z]+)*");
    static final List<String> KEYS = Arrays.asList("seconds", "context", "lr", "data", "chars", "valchars", "valstart", "name");

    static Map<String, String> opt = new HashMap<>();
    static int intOpt(String k, int dflt) { return opt.containsKey(k) ? Integer.parseInt(opt.get(k)) : dflt; }

    // usage:  java ChatTransformer [key=value ...]        train, then chat
    //   keys: seconds context lr data chars valchars valstart name   (old style also works: java ChatTransformer 420 10 0.001)
    //   valstart=N takes the held-out text from position N of the file instead of right after the training text,
    //   so runs with different amounts of training text can be scored on exactly the same held-out passage.
    //         java ChatTransformer chat [name=...]          chat with a saved model, no training
    // name=tiny saves tiny-best.bin and tiny-last.bin, so models trained on different data do not overwrite each other.
    public static void main(String[] args) throws IOException {
        boolean chatOnly = false;
        List<String> plain = new ArrayList<>();
        for (String a : args) {
            if (a.equals("chat")) chatOnly = true;
            else if (a.contains("=")) {
                String[] kv = a.split("=", 2);
                if (!KEYS.contains(kv[0])) { System.out.println("Unknown option '" + kv[0] + "'. Valid options: " + KEYS); return; }
                opt.put(kv[0], kv[1]);
            } else plain.add(a);
        }
        String[] positional = { "seconds", "context", "lr" };
        for (int i = 0; i < plain.size() && i < positional.length; i++) opt.putIfAbsent(positional[i], plain.get(i));

        String prefix = opt.containsKey("name") ? opt.get("name") + "-" : "";
        String best = prefix + "best.bin", last = prefix + "last.bin";
        if (chatOnly) {
            chat(MultiLayerTransformer.load(best), "saved " + best);
            return;
        }

        int maxSeconds = intOpt("seconds", DEFAULT_SECONDS);
        int T = intOpt("context", DEFAULT_CONTEXT);
        double lr = opt.containsKey("lr") ? Double.parseDouble(opt.get("lr")) : LEARNING_RATE;
        String dataFile = opt.getOrDefault("data", DEFAULT_DATA);
        int valChars = intOpt("valchars", DEFAULT_VAL_CHARS);

        String all = clean(new String(Files.readAllBytes(Paths.get(dataFile)), StandardCharsets.UTF_8));
        int valStart = opt.containsKey("valstart") ? intOpt("valstart", 0) : -1;
        int trainChars = Math.min(intOpt("chars", DEFAULT_CHARS), valStart >= 0 ? valStart : all.length() - valChars);
        if (trainChars < 2 * valChars || (valStart >= 0 && valStart + valChars > all.length())) { System.out.println("Not enough text in " + dataFile + " (" + all.length() + " characters after cleaning) for these settings."); return; }
        String train = all.substring(0, trainChars);
        if (valStart < 0) valStart = trainChars;
        String val = all.substring(valStart, valStart + valChars);
        String probe = train.substring(trainChars / 2, trainChars / 2 + valChars);   // training text, measured the same way as val
        Set<String> realWords = new HashSet<>();
        Matcher wm = WORD.matcher(all.toLowerCase());
        while (wm.find()) realWords.add(wm.group());

        String settings = String.format("data=%s chars=%d val@%d context=%d dim=%d layers=%d heads=%d lr=%s", dataFile, trainChars, valStart, T, DIM, LAYERS, HEADS, lr);
        System.out.printf("Training for up to %d seconds (%s).%n", maxSeconds, settings);
        MultiLayerTransformer model = new MultiLayerTransformer(train, T, DIM, LAYERS, HEADS, lr);
        System.out.printf("Vocabulary: %d characters. Before training: val loss %.3f (pure guessing = %.2f; lower is better)%n",
            model.vocabSize, model.evaluate(val, 1000), Math.log(model.vocabSize));
        System.out.println("  passes = how many times the text's characters have been used as prediction targets");

        Random rng = new Random(SEED);
        long start = System.currentTimeMillis(), nextReport = start + REPORT_SECONDS * 1000L;
        long steps = 0;
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
                if (v < bestVal) { bestVal = v; bestSeconds = (int) ((now - start) / 1000); model.save(best); }
                runLoss = 0; runN = 0;
                nextReport = System.currentTimeMillis() + REPORT_SECONDS * 1000L;
            }
        }
        model.save(last);
        double finalVal = model.evaluate(val, 1000), finalTrain = model.evaluate(probe, 1000);
        System.out.printf("Done. %,d steps. Final val %.3f (train-eval %.3f). Best val %.3f at %ds, saved to %s.%n",
            steps, finalVal, finalTrain, bestVal, bestSeconds, best);
        String sample = generate(model, val.substring(0, T), 300, new Random(99));
        System.out.println("Sample (temperature " + TEMPERATURE + "):\n" + sample + "\n");

        String line = String.format("%s | %s | %ds %,d steps | best val %.3f @%ds | final val %.3f | train-eval %.3f | real words %.0f%% | copied %.0f%%%n",
            LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")), settings, maxSeconds, steps,
            bestVal, bestSeconds, finalVal, finalTrain, st[0], st[1]);
        Files.writeString(Paths.get(LOG), line, StandardOpenOption.CREATE, StandardOpenOption.APPEND);

        chat(MultiLayerTransformer.load(best), String.format("best checkpoint (val %.3f)", bestVal));
    }

    /** Tidies any text file: story separators become newlines, fancy quotes and dashes become plain ones, very rare characters are dropped. */
    static String clean(String raw) {
        String t = raw.replace("<|endoftext|>", "\n").replace("\r", "")
            .replace((char) 0x2018, '\'').replace((char) 0x2019, '\'')     // curly single quotes
            .replace((char) 0x201C, '"').replace((char) 0x201D, '"')        // curly double quotes
            .replace((char) 0x2013, '-').replace((char) 0x2014, '-')        // dashes
            .replace(String.valueOf((char) 0x2026), "...").replace((char) 0xA0, ' ');
        Map<Character, Integer> counts = new HashMap<>();
        for (int i = 0; i < t.length(); i++) counts.merge(t.charAt(i), 1, Integer::sum);
        StringBuilder sb = new StringBuilder(t.length());
        for (int i = 0; i < t.length(); i++) if (counts.get(t.charAt(i)) >= MIN_CHAR_COUNT) sb.append(t.charAt(i));
        return sb.toString();
    }

    static String generate(MultiLayerTransformer model, String seed, int length, Random rand) {
        StringBuilder out = new StringBuilder(seed);
        for (int i = 0; i < length; i++) {
            out.append(model.sampleNext(out.substring(Math.max(0, out.length() - model.contextLength)), rand, TEMPERATURE));
        }
        return out.toString();
    }

    /** {percent of generated words that appear somewhere in the data file, percent of generated 20-character pieces copied verbatim from the training text} */
    static double[] textStats(MultiLayerTransformer model, String seed, Set<String> realWords, String train) {
        String text = generate(model, seed, 1000, new Random(99)).substring(seed.length());
        List<String> words = new ArrayList<>();
        Matcher m = WORD.matcher(text.toLowerCase());
        while (m.find()) words.add(m.group());
        if (words.size() > 2) words = words.subList(1, words.size() - 1);   // first and last may be cut in half
        int hits = 0;
        for (String w : words) if (realWords.contains(w)) hits++;
        int copied = 0, checked = 0;
        for (int i = 0; i + 20 <= text.length(); i += 5) {                  // every 5th piece is enough and keeps this fast on big files
            checked++;
            if (train.contains(text.substring(i, i + 20))) copied++;
        }
        return new double[] { words.isEmpty() ? 0 : 100.0 * hits / words.size(), checked == 0 ? 0 : 100.0 * copied / checked };
    }

    static void chat(MultiLayerTransformer model, String which) {
        System.out.println("Chatting with the " + which + ". Note: this model CONTINUES text in the style of its training data; it cannot answer questions yet.");
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
