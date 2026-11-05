import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Step 26: does the model really use the user's message when it writes a reply?
 *
 * For every held-out (user line, bot reply) pair it measures how well the model predicts the REAL reply
 *   (a) after the real user line, and
 *   (b) after a different, randomly chosen user line (same earlier lines, only the user line is swapped).
 * A model that listens to the question predicts the reply better in (a) than in (b). A model that only
 * writes plausible chit-chat predicts both about equally, so the gain is close to zero.
 * Both numbers are in nats per character of the reply. The model is not changed or trained.
 *
 * Usage: java ReplyCheck name=chat data=chitchat.txt chars=5573713 [pairs=400] [shuffles=3] [seed=1]
 *        (chars = the training characters in the data file; everything after them is held-out text)
 */
public class ReplyCheck {

    static final int MAX_CONTEXT_LINES = 6;

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = new HashMap<>();
        opt.put("name", "chat");
        opt.put("data", "chitchat.txt");
        opt.put("pairs", "400");
        opt.put("shuffles", "3");
        opt.put("seed", "1");
        Set<String> known = new HashSet<>(Arrays.asList("name", "data", "chars", "pairs", "shuffles", "seed"));
        for (String a : args) {
            int eq = a.indexOf('=');
            if (eq <= 0 || !known.contains(a.substring(0, eq))) {
                System.out.println("Unknown argument: " + a + "\nKnown: " + known);
                return;
            }
            opt.put(a.substring(0, eq), a.substring(eq + 1));
        }
        if (!opt.containsKey("chars")) {
            System.out.println("chars= is needed (the number of training characters, printed by ChitChatData).");
            return;
        }
        String text = new String(Files.readAllBytes(Paths.get(opt.get("data"))), StandardCharsets.UTF_8);
        int chars = Integer.parseInt(opt.get("chars"));
        if (chars <= 0 || chars >= text.length()) {
            System.out.println("chars=" + chars + " does not fit a file of " + text.length() + " characters.");
            return;
        }
        String file = opt.get("name") + "-best.bin";
        MultiLayerTransformer model = MultiLayerTransformer.load(file);
        System.out.printf("Loaded %s: %,d parameters, sees %d tokens at once.%n", file, model.parameterCount(), model.contextLength);
        check(model, text.substring(chars), Integer.parseInt(opt.get("pairs")), Integer.parseInt(opt.get("shuffles")),
              Long.parseLong(opt.get("seed")), true);
    }

    /** One user line followed by the bot's reply, with the lines said before them in that conversation. */
    static final class Pair {
        final List<String> before = new ArrayList<>();
        String user, bot;
        boolean opening;
    }

    static List<Pair> pairsOf(String heldOut) {
        List<Pair> out = new ArrayList<>();
        for (String block : heldOut.split("=====")) {
            List<String> lines = new ArrayList<>();
            for (String l : block.split("\n")) {
                l = l.trim();
                if (l.startsWith("User: ") || l.startsWith("Bot: ")) lines.add(l);
            }
            for (int i = 0; i + 1 < lines.size(); i++) {
                if (!lines.get(i).startsWith("User: ") || !lines.get(i + 1).startsWith("Bot: ")) continue;
                Pair p = new Pair();
                for (int j = Math.max(0, i - MAX_CONTEXT_LINES); j < i; j++) p.before.add(lines.get(j));
                p.user = lines.get(i).substring(6);
                p.bot = lines.get(i + 1).substring(5);
                p.opening = i == 0;
                out.add(p);
            }
        }
        return out;
    }

    static double nll(MultiLayerTransformer m, int[] ids) { return m.sequenceLoss(ids) * (ids.length - 1); }

    /**
     * Total loss (nats) of the reply tokens given everything before them, or NaN when the prefix and the reply do not
     * split cleanly into tokens or when everything does not fit into the model's window. The loss of the whole window
     * minus the loss of the prefix alone is exactly the loss of the reply tokens, because the model only looks backwards.
     */
    static double replyNll(MultiLayerTransformer m, String prefix, String reply) {
        int[] p = m.tokenizer.encode(prefix);
        int[] f = m.tokenizer.encode(prefix + reply);
        if (p.length < 2 || f.length <= p.length || f.length > m.contextLength + 1) return Double.NaN;
        for (int i = 0; i < p.length; i++) if (f[i] != p[i]) return Double.NaN;
        return nll(m, f) - nll(m, p);
    }

    static String prefixOf(List<String> before, int keep, String user) {
        StringBuilder sb = new StringBuilder();
        for (int j = before.size() - keep; j < before.size(); j++) sb.append(before.get(j)).append('\n');
        return sb.append("User: ").append(user).append("\nBot:").toString();
    }

    static void check(MultiLayerTransformer m, String heldOut, int wanted, int shuffles, long seed, boolean verbose) {
        List<Pair> all = pairsOf(heldOut);
        if (all.size() < 2) { System.out.println("Fewer than 2 held-out pairs found; nothing to check."); return; }
        Random rand = new Random(seed);
        List<Pair> order = new ArrayList<>(all);
        Collections.shuffle(order, rand);

        List<Double> realAll = new ArrayList<>(), shufAll = new ArrayList<>();
        List<Double> realOpen = new ArrayList<>(), shufOpen = new ArrayList<>();
        List<String> shown = new ArrayList<>();
        List<double[]> gains = new ArrayList<>();      // {gain per character, index into `shown`}
        int skipped = 0, scored = 0;
        for (Pair p : order) {
            if (scored >= wanted) break;
            String reply = " " + p.bot + "\n";
            int chars = m.tokenizer.filter(reply).length();
            // keep as many earlier lines as still fit into the window
            int keep = -1;
            double real = Double.NaN;
            for (int k = p.before.size(); k >= 0; k--) {
                real = replyNll(m, prefixOf(p.before, k, p.user), reply);
                if (!Double.isNaN(real)) { keep = k; break; }
            }
            if (keep < 0) { skipped++; continue; }
            double shufSum = 0;
            int trials = 0;
            for (int t = 0, tries = 0; t < shuffles && tries < 50; tries++) {
                Pair other = all.get(rand.nextInt(all.size()));
                if (other.user.equals(p.user)) continue;
                double s = replyNll(m, prefixOf(p.before, keep, other.user), reply);
                if (Double.isNaN(s)) continue;
                shufSum += s / chars;
                trials++;
                t++;
            }
            if (trials == 0) { skipped++; continue; }
            double r = real / chars, s = shufSum / trials;
            realAll.add(r); shufAll.add(s);
            if (p.opening) { realOpen.add(r); shufOpen.add(s); }
            shown.add("User: " + p.user + "  ->  Bot:" + reply.trim().replace("\n", " ") + String.format("   (real %.2f, random %.2f)", r, s));
            gains.add(new double[] { s - r, shown.size() - 1 });
            scored++;
        }
        System.out.printf("Scored %,d held-out (user line, bot reply) pairs; skipped %,d that did not fit into %d tokens. Scores are nats per character of the reply.%n",
            scored, skipped, m.contextLength);
        if (scored < 20) { System.out.println("Too few pairs for a meaningful result."); return; }
        report("All pairs", realAll, shufAll);
        if (realOpen.size() >= 20) report("Opening exchanges only (nothing said before)", realOpen, shufOpen);
        else System.out.printf("Opening exchanges: only %d pairs, too few to report separately.%n", realOpen.size());
        System.out.println("A gain near zero means the model ignores the question and just writes plausible chit-chat.");

        if (verbose) {
            gains.sort((a, b) -> Double.compare(b[0], a[0]));
            System.out.println("\nPairs where the real question helped most:");
            for (int i = 0; i < Math.min(3, gains.size()); i++) System.out.println("  " + shown.get((int) gains.get(i)[1]));
            System.out.println("Pairs where it helped least (or hurt):");
            for (int i = Math.max(0, gains.size() - 3); i < gains.size(); i++) System.out.println("  " + shown.get((int) gains.get(i)[1]));
        }
    }

    static void report(String title, List<Double> real, List<Double> shuf) {
        int n = real.size();
        double mr = 0, ms = 0;
        for (int i = 0; i < n; i++) { mr += real.get(i); ms += shuf.get(i); }
        mr /= n; ms /= n;
        double var = 0;
        int wins = 0;
        for (int i = 0; i < n; i++) {
            double g = shuf.get(i) - real.get(i);
            var += (g - (ms - mr)) * (g - (ms - mr));
            if (g > 0) wins++;
        }
        double se = Math.sqrt(var / Math.max(1, n - 1) / n);
        System.out.printf("%s (%d pairs):%n", title, n);
        System.out.printf("  reply loss with the real question:   %.3f%n", mr);
        System.out.printf("  reply loss with a random question:   %.3f%n", ms);
        System.out.printf("  gain from the real question: %.3f +/- %.3f (standard error), a %.0f%% lower loss; the real question wins in %.0f%% of pairs%n",
            ms - mr, se, 100.0 * (ms - mr) / ms, 100.0 * wins / n);
    }
}
