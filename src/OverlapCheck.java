import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Baseline for the "copied" number in the training report.
 * The report counts how many 20-character pieces of generated text appear word for word in the training text.
 * That number only means something next to this one: how many 20-character pieces of REAL text the model never trained on
 * appear word for word in the training text (stories reuse stock phrases, so it is not zero).
 *
 * Usage: java OverlapCheck [datafile] [trainchars] [testchars]     (defaults: tinystories.txt 2500000 50000)
 * Use the same datafile and trainchars as the training run you want to judge. Nothing is written or changed.
 */
public class OverlapCheck {
    static final int PIECE = 20, STEP = 5, CHUNK = 1000;   // the same piece size and spacing as the training report

    static double share(String text, String train) {
        int found = 0, checked = 0;
        for (int i = 0; i + PIECE <= text.length(); i += STEP) {
            checked++;
            if (train.contains(text.substring(i, i + PIECE))) found++;
        }
        return checked == 0 ? 0 : 100.0 * found / checked;
    }

    public static void main(String[] args) throws Exception {
        String file = args.length > 0 ? args[0] : "tinystories.txt";
        int trainChars = args.length > 1 ? Integer.parseInt(args[1]) : 2500000;
        int testChars = args.length > 2 ? Integer.parseInt(args[2]) : 50000;
        String all = ChatTransformer.clean(new String(Files.readAllBytes(Paths.get(file)), StandardCharsets.UTF_8));
        if (all.length() < trainChars + CHUNK) { System.out.println("Not enough text: " + all.length() + " characters after cleaning, need more than " + (trainChars + CHUNK) + "."); return; }
        String train = all.substring(0, trainChars);
        String unseen = all.substring(trainChars, Math.min(all.length(), trainChars + testChars));
        long t0 = System.currentTimeMillis();

        double sanity = share(train.substring(trainChars / 2, trainChars / 2 + Math.min(10000, trainChars / 2)), train);
        System.out.printf("Sanity check: pieces taken from the training text itself are found %.0f%% of the time (must be 100%%).%n", sanity);

        List<Double> perChunk = new ArrayList<>();
        for (int s = 0; s + CHUNK <= unseen.length(); s += CHUNK) perChunk.add(share(unseen.substring(s, s + CHUNK), train));
        Collections.sort(perChunk);
        System.out.printf("Real text the model never trained on (%,d characters, %d samples of %d characters, like the 1,000-character generated sample):%n", unseen.length(), perChunk.size(), CHUNK);
        System.out.printf("  whole passage: %.1f%% of 20-character pieces appear word for word in the training text%n", share(unseen, train));
        System.out.printf("  per sample: lowest %.1f%%, median %.1f%%, highest %.1f%%%n", perChunk.get(0), perChunk.get(perChunk.size() / 2), perChunk.get(perChunk.size() - 1));
        System.out.printf("(%.1fs)%n", (System.currentTimeMillis() - t0) / 1000.0);
        System.out.println("Compare with the 'copied' column: a generated sample far above these numbers would point to memorization; similar numbers mean the model repeats stock phrases about as often as real stories do.");
    }
}
