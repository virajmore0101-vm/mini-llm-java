import java.nio.file.*;

public class Benchmark {
    public static void main(String[] args) throws Exception {
        String text = Files.readString(Paths.get("shakespeare.txt")).substring(0, 3000);
        int T = 10;
        MultiLayerTransformer model = new MultiLayerTransformer(text, T, 24, 3, 4, 0.004);
        long start = System.currentTimeMillis();
        double loss = 0;
        for (int epoch = 0; epoch < 3; epoch++) {
            double total = 0;
            int count = 0;
            for (int i = 0; i + T < text.length(); i++) {
                total += model.trainStep(text.substring(i, i + T), text.charAt(i + T));
                count++;
            }
            loss = total / count;
        }
        long ms = System.currentTimeMillis() - start;
        System.out.printf("Final loss: %.8f | time: %.1fs | %.0f steps/sec%n", loss, ms / 1000.0, 3.0 * (text.length() - T) / (ms / 1000.0));
    }
}
