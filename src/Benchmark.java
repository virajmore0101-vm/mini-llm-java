import java.nio.file.*;
import java.util.Arrays;
import java.util.Random;

/** Measures training speed: java Benchmark [dim] [batch] [threads] [seconds] [vocab]   (defaults: 64 16 all-cores 15 0; vocab 0 = characters) */
public class Benchmark {
    public static void main(String[] args) throws Exception {
        int dim = args.length > 0 ? Integer.parseInt(args[0]) : 64;
        int batch = args.length > 1 ? Integer.parseInt(args[1]) : 16;
        int threads = args.length > 2 ? Integer.parseInt(args[2]) : Runtime.getRuntime().availableProcessors();
        double seconds = args.length > 3 ? Double.parseDouble(args[3]) : 15;
        int vocab = args.length > 4 ? Integer.parseInt(args[4]) : 0;
        int T = 32;
        String text = Files.readString(Paths.get("shakespeare.txt")).substring(0, 200000);
        Tokenizer tok = vocab > 0 ? Tokenizer.train(text, vocab) : Tokenizer.chars(text);
        int[] ids = tok.encode(text);
        MultiLayerTransformer model = new MultiLayerTransformer(tok, T, dim, 3, 4, 0.001);
        model.setThreads(threads);
        Random rand = new Random(5);
        int[][] windows = new int[batch][];
        long begin = System.currentTimeMillis(), measureFrom = -1, windowsAtStart = 0, done = 0;
        while (System.currentTimeMillis() - begin < (seconds + 2) * 1000) {
            for (int i = 0; i < batch; i++) { int s = rand.nextInt(ids.length - T - 1); windows[i] = Arrays.copyOfRange(ids, s, s + T + 1); }
            model.trainBatch(windows);
            done += batch;
            if (measureFrom < 0 && System.currentTimeMillis() - begin > 2000) { measureFrom = System.currentTimeMillis(); windowsAtStart = done; }   // first 2 seconds are warm-up
        }
        double rate = (done - windowsAtStart) / ((System.currentTimeMillis() - measureFrom) / 1000.0);
        System.out.printf("dim=%d vocab=%s (%,d parameters) batch=%d threads=%d: %,.0f windows per second%n",
            dim, vocab > 0 ? String.valueOf(tok.size()) : "chars", model.parameterCount(), batch, threads, rate);
    }
}
