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
    static final int DEFAULT_DIM = 64;          // width of the model; 24 was the old size (24,000 parameters), 64 is about 156,000
    static final int DEFAULT_LAYERS = 3;
    static final int DEFAULT_HEADS = 4;         // dim must be divisible by heads
    static final int DEFAULT_BATCH = 16;        // windows per update: their gradients are averaged, then one Adam step is made
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
    static final List<String> KEYS = Arrays.asList("seconds", "context", "lr", "data", "chars", "valchars", "valstart", "name", "dim", "layers", "heads", "batch", "threads", "windows", "vocab");

    static Map<String, String> opt = new HashMap<>();
    static int intOpt(String k, int dflt) { return opt.containsKey(k) ? Integer.parseInt(opt.get(k)) : dflt; }

    // usage:  java ChatTransformer [key=value ...]        train, then chat
    //   keys: seconds context lr data chars valchars valstart name dim layers heads batch threads   (old style also works: java ChatTransformer 420 10 0.001)
    //   dim/layers/heads set the model size; batch = windows per update; threads = CPU threads to use (default: all cores).
    //   vocab=N sets how the text is cut into tokens: 0 (default) = one token per character; e.g. vocab=512 = about 440 learned
    //   word-pieces (frequent words and word-parts such as " the" or "ing") on top of the characters, so a window covers much more text.
    //   Losses are always reported PER CHARACTER, so runs with different vocab settings can be compared on the same held-out passage.
    //   dim=24 batch=1 threads=1 is the same model and training as Step 21 (except the learning rate is now lowered after every update, not every 128).
    //   windows=N trains for exactly N windows instead of for `seconds` (the learning rate shrinks over those N windows), so runs
    //   on machines of different speed, or with different model sizes, do the same amount of training and can be compared fairly.
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
        int dim = intOpt("dim", DEFAULT_DIM), layers = intOpt("layers", DEFAULT_LAYERS), heads = intOpt("heads", DEFAULT_HEADS);
        int batch = intOpt("batch", DEFAULT_BATCH);
        long maxWindows = opt.containsKey("windows") ? Long.parseLong(opt.get("windows")) : 0;
        int vocab = intOpt("vocab", 0);
        int threads = intOpt("threads", Runtime.getRuntime().availableProcessors());
        if (dim % heads != 0 || batch < 1 || threads < 1 || layers < 1 || vocab < 0) { System.out.println("Need dim divisible by heads, and batch, threads, layers >= 1 and vocab >= 0."); return; }

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

        String settings = String.format("data=%s chars=%d val@%d context=%d dim=%d layers=%d heads=%d batch=%d threads=%d lr=%s vocab=%s%s", dataFile, trainChars, valStart, T, dim, layers, heads, batch, threads, lr, vocab > 0 ? String.valueOf(vocab) : "chars", maxWindows > 0 ? " windows=" + maxWindows : "");
        System.out.printf("Training for %s (%s).%n", maxWindows > 0 ? String.format("%,d windows", maxWindows) : "up to " + maxSeconds + " seconds", settings);
        long tokenizerStart = System.currentTimeMillis();
        Tokenizer tok = vocab > 0 ? Tokenizer.train(train, vocab) : Tokenizer.chars(train);
        int[] trainIds = tok.encode(train);
        double charsPerToken = (double) train.length() / trainIds.length;
        System.out.printf("Tokenizer: %d tokens (%d characters + %d learned word-pieces), %.2f characters per token on average (built in %.1fs). A window of %d tokens covers about %.0f characters.%n",
            tok.size(), tok.size() - tok.numMerges(), tok.numMerges(), charsPerToken, (System.currentTimeMillis() - tokenizerStart) / 1000.0, T + 1, (T + 1) * charsPerToken);
        MultiLayerTransformer model = new MultiLayerTransformer(tok, T, dim, layers, heads, lr);
        model.setThreads(threads);
        System.out.printf("Model: %,d parameters. Before training: val loss %.3f per character (pure guessing = %.2f; lower is better)%n",
            model.parameterCount(), model.evaluate(val, 1000), Math.log(tok.size()) / charsPerToken);
        System.out.println("  windows = training snippets processed; passes = how many times the text has been used as prediction targets; all losses are per character");

        Random rng = new Random(SEED);
        long start = System.currentTimeMillis(), nextReport = start + REPORT_SECONDS * 1000L;
        long steps = 0;                 // windows processed (one update uses `batch` of them)
        int[][] windows = new int[batch][];
        double runLoss = 0; int runN = 0, bestSeconds = 0;
        double bestVal = Double.MAX_VALUE;
        double[] st = {0, 0};
        boolean done = false;
        while (!done) {
            for (int i = 0; i < batch; i++) {
                int s = rng.nextInt(trainIds.length - T - 1);               // random windows, so no part of the text is favored
                windows[i] = Arrays.copyOfRange(trainIds, s, s + T + 1);
            }
            runLoss += model.trainBatch(windows);
            runN++; steps += batch;
            long now = System.currentTimeMillis();
            double progress = maxWindows > 0 ? (double) steps / maxWindows : (now - start) / (maxSeconds * 1000.0);
            done = progress >= 1.0;
            model.learningRate = lr * (1 - (1 - LR_END_FRACTION) * Math.min(1.0, progress));
            if (now >= nextReport || done) {
                double v = model.evaluate(val, 1000);
                st = textStats(model, val.substring(0, T), realWords, train);
                System.out.printf("%4ds | %,9d windows | %5.1f passes | train %.3f | train-eval %.3f | val %.3f | real words %2.0f%% | copied %2.0f%%%n",
                    (now - start) / 1000, steps, (double) steps * T / trainIds.length, runLoss / runN / charsPerToken,
                    model.evaluate(probe, 1000), v, st[0], st[1]);
                if (v < bestVal) { bestVal = v; bestSeconds = (int) ((now - start) / 1000); model.save(best); }
                runLoss = 0; runN = 0;
                nextReport = System.currentTimeMillis() + REPORT_SECONDS * 1000L;
            }
        }
        model.save(last);
        double finalVal = model.evaluate(val, 1000), finalTrain = model.evaluate(probe, 1000);
        System.out.printf("Done. %,d windows (%,.0f per second). Final val %.3f (train-eval %.3f). Best val %.3f at %ds, saved to %s.%n",
            steps, steps / ((System.currentTimeMillis() - start) / 1000.0), finalVal, finalTrain, bestVal, bestSeconds, best);
        String sample = generate(model, val.substring(0, T), 300, new Random(99));
        System.out.println("Sample (temperature " + TEMPERATURE + "):\n" + sample + "\n");

        String line = String.format("%s | %s | %ds %,d windows | %,d params | best val %.3f @%ds | final val %.3f | train-eval %.3f | real words %.0f%% | copied %.0f%%%n",
            LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")), settings, (System.currentTimeMillis() - start) / 1000, steps, model.parameterCount(),
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

    /** The seed text followed by `length` generated characters. */
    static String generate(MultiLayerTransformer model, String seed, int length, Random rand) {
        return model.tokenizer.filter(seed) + model.generate(seed, length, rand, TEMPERATURE);
    }

    /** {percent of generated words that appear somewhere in the data file, percent of generated 20-character pieces copied verbatim from the training text} */
    static double[] textStats(MultiLayerTransformer model, String seed, Set<String> realWords, String train) {
        String text = model.generate(seed, 1000, new Random(99), TEMPERATURE);
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

            String prompt = model.tokenizer.filter(input);
            int skipped = input.length() - prompt.length();
            if (skipped > 0) System.out.println("(Skipped " + skipped + " character(s) the model has never seen.)");
            if (prompt.isEmpty()) prompt = model.tokenizer.knows('\n') ? "\n" : model.tokenizer.tokenText(0);
            System.out.println("\nModel: " + prompt + model.generate(prompt, REPLY_LENGTH, rand, TEMPERATURE) + "\n");
        }
        scanner.close();
    }
}
