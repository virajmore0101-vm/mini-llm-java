import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Scanner;

public class ChatTransformer {
    public static void main(String[] args) throws IOException {
        String text = Files.readString(Paths.get("shakespeare.txt")).substring(0, 3000);
        int T = 10;

        System.out.println("Training the scaled-up transformer (3 layers, 4 heads, 24-dim) on "
            + text.length() + " characters... this takes about 4-5 minutes, not a hang.");
        MultiLayerTransformer model = new MultiLayerTransformer(text, T, 24, 3, 4, 0.004);
        long start = System.currentTimeMillis();
        for (int epoch = 0; epoch < 100; epoch++) {
            for (int i = 0; i + T < text.length(); i++) {
                model.trainStep(text.substring(i, i + T), text.charAt(i + T));
            }
            if (epoch % 20 == 0) {
                System.out.println("  ...epoch " + epoch + " (" + (System.currentTimeMillis() - start) / 1000 + "s elapsed)");
            }
        }
        System.out.println("Ready. Type something and press enter (or 'exit' to quit):\n");

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

            String seed = findSeed(model, text, input, T, rand);
            StringBuilder out = new StringBuilder(seed);
            for (int i = 0; i < 200; i++) {
                out.append(model.sampleNext(out.substring(out.length() - T), rand));
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
        System.out.println("(Your exact text wasn't all in the tiny training vocabulary -- starting from a real snippet instead.)");
        int randomStart = rand.nextInt(text.length() - T);
        return text.substring(randomStart, randomStart + T);
    }
}
