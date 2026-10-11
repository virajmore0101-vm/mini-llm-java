import java.nio.file.*;
import java.util.Random;

/** Measures training speed: java Benchmark [dim] [batch] [threads] [seconds]   (defaults: 64 16 all-cores 15) */
public class Benchmark {
    public static void main(String[] args) throws Exception {
        int dim = args.length > 0 ? Integer.parseInt(args[0]) : 64;
        int batch = args.length > 1 ? Integer.parseInt(args[1]) : 16;
        int threads = args.length > 2 ? Integer.parseInt(args[2]) : Runtime.getRuntime().availableProcessors();
        double seconds = args.length > 3 ? Double.parseDouble(args[3]) : 15;
        int T = 32;
        String text = Files.readString(Paths.get("shakespeare.txt")).substring(0, 200000);
        MultiLayerTransformer model = new MultiLayerTransformer(text, T, dim, 3, 4, 0.001);
        model.setThreads(threads);
        Random rand = new Random(5);
        String[] windows = new String[batch];
        long begin = System.currentTimeMillis(), measureFrom = -1, windowsAtStart = 0, done = 0;
        while (System.currentTimeMillis() - begin < (seconds + 2) * 1000) {
            for (int i = 0; i < batch; i++) { int s = rand.nextInt(text.length() - T - 1); windows[i] = text.substring(s, s + T + 1); }
            model.trainBatch(windows);
            done += batch;
            if (measureFrom < 0 && System.currentTimeMillis() - begin > 2000) { measureFrom = System.currentTimeMillis(); windowsAtStart = done; }   // first 2 seconds are warm-up
        }
        double rate = (done - windowsAtStart) / ((System.currentTimeMillis() - measureFrom) / 1000.0);
        System.out.printf("dim=%d (%,d parameters) batch=%d threads=%d: %,.0f windows per second%n", dim, model.parameterCount(), batch, threads, rate);
    }
}
